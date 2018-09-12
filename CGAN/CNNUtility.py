# CNNUtility.py
# Written Ian Rankin September 2018
#
# Sets of utility functions for generating CNN's


import tensorflow as tf


########################### Define the batch norm function.



# batchNormLayer
# Adds a batch normalization layer to to the filter.
# x - input tensor
# filterShape - shape of filter
# num - the number to not have the same variable name for the gamma and beta variables.
# filtType - the type of filter (conv, mult)
def batchNormLayer(x, numChannels, num, trainPhase, filtType='conv', decayRate = 0.99):
    # define the batch norm function for use.
    betaInit = tf.zeros_initializer(dtype=tf.float32)
    gammaInit = tf.ones_initializer(dtype=tf.float32)

    # assumed to be convlution filter

    #define weight variables
    gamma = tf.get_variable('gamma' + str(num), [numChannels], initializer=gammaInit)
    beta = tf.get_variable('beta' + str(num), [numChannels], initializer=betaInit)

    axes = []
    if filtType == 'mult':
        axes = [0]
    else:
        axes = [0,1,2]

    batch_mean, batch_variance = tf.nn.moments(x, axes)

    ema = tf.train.ExponentialMovingAverage(decay=decayRate)

    def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_variance])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

    mean, variance = tf.cond(trainPhase,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_variance)))



    normed = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.000001)
    return normed, gamma, beta, ema.average(batch_mean), ema.average(batch_variance)

####################### define conv filters.

numLayers = 0
# Define variable initilization
normInit = tf.truncated_normal_initializer(0, .05, dtype=tf.float32)
zeroInit = tf.constant_initializer(0.0, dtype=tf.float32)

# convLayer
# define convolutional layer with batch normalization, and max pooling
# @param x - the input tensor
# @param filterShape - the shape of the filter (height, width, num channels output)
# @param poolShape - the shape of the pooling (height width)
#
# @return layer, list of all variables
def convLayer(x, filterShape, poolShape, trainPhase):
    global numLayers
    inputChannels = x.shape[3]
    convFilt = tf.get_variable('filt' + str(numLayers), \
        [filterShape[0], filterShape[1], inputChannels, filterShape[2]], \
        initializer=normInit)
    bias = tf.get_variable('bias' + str(numLayers), \
        [filterShape[2]], initializer=zeroInit)

    # setup layer conv, batch norm, and pooling layers.
    logit = tf.nn.conv2d(x, convFilt, strides=[1,1,1,1], padding='SAME') + bias
    normed, gamma, beta, mean, variance = \
        batchNormLayer(logit, filterShape[2], numLayers, trainPhase)
    layer = tf.nn.relu(normed)
    pooled = tf.nn.max_pool(layer, \
                ksize=[1,poolShape[0], poolShape[1], 1], \
                strides=[1,poolShape[0], poolShape[1], 1], \
                padding='SAME')


    numLayers += 1
    return pooled, [convFilt, bias, gamma, beta], [mean, variance]


# transposeConvLayer
# Define a transposed convolutional layer with batch normalization
# @param x - the input tensor
# @param filterShape - the shape of the filter (height, width, num channels output)
# @param outputShape - the shape of the output image (height width)
#
# @return layer, list of all variables
def transposeConvLayer(x, filterShape, outShape, strides, trainPhase):
    global numLayers
    inputChannels = x.shape[3]
    convFilt = tf.get_variable('filt' + str(numLayers), \
        [filterShape[0], filterShape[1], filterShape[2], inputChannels], \
        initializer=normInit)
    bias = tf.get_variable('bias' + str(numLayers), \
        [filterShape[2]], initializer=zeroInit)

    # setup layer conv, batch norm, and pooling layers.
    print(x.shape[0])
    logit = tf.nn.conv2d_transpose(x, convFilt,\
        output_shape=[tf.shape(x)[0],outShape[0],outShape[1], filterShape[2]],\
        strides=strides, padding='SAME') + bias
    normed, gamma, beta, mean, variance = \
        batchNormLayer(logit, filterShape[2], numLayers, trainPhase)
    layer = tf.nn.relu(normed)


    numLayers += 1
    return layer, [convFilt, bias, gamma, beta], [mean, variance]


# fullConnLayer
# Define a fully connected layer with batch normalization
# @param x - the input tensor
# @param numOutputNodes - number of output nodes
#
# @return outputLayer, list of all variables
def fullConnLayer(x, numOutputNodes, trainPhase):
    global numLayers
    inputChannels = x.shape[1]

    matFilt = tf.get_variable('filt' + str(numLayers), \
        [inputChannels, numOutputNodes], initializer=normInit)
    bias = tf.get_variable('bias' + str(numLayers), \
        [numOutputNodes], initializer=zeroInit)

    logit = tf.matmul(x, matFilt) + bias
    normed, gamma, beta, mean, variance = \
        batchNormLayer(logit, numOutputNodes, numLayers, trainPhase, 'mult')
    layer = tf.nn.relu(normed)

    numLayers += 1
    return layer, [matFilt, bias, gamma, beta], [mean, variance]

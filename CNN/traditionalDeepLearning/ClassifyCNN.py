# ClassifyCNN.py
# Written Ian Rankin September 2018
#
# This code is adopted from the ipynotebook
# to allow the code to run on the NMSU supercomputer

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

imageShape = (405,720,3)
print(imageShape)

###########################
# setup the place holder input for the images
imageWidth = imageShape[1]
imageHeight = imageShape[0]

x = tf.placeholder(tf.float32, shape=[None, imageHeight, imageWidth, imageShape[2]])
#xWhite = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)

# setup placeholder input for labels
y = tf.placeholder(tf.float32, shape=[None, 2])

# placeholder for batch norm training phase.
trainPhase = tf.placeholder(tf.bool)

batchSize = 30
print(x)
#print(xWhite)
print(y)

########################### Define the batch norm function.

# define the batch norm function for use.
decayRate = 0.98
betaInit = tf.zeros_initializer(dtype=tf.float32)
gammaInit = tf.ones_initializer(dtype=tf.float32)

# batchNormLayer
# Adds a batch normalization layer to to the filter.
# x - input tensor
# filterShape - shape of filter
# num - the number to not have the same variable name for the gamma and beta variables.
# filtType - the type of filter (conv, mult)
def batchNormLayer(x, numChannels, num, filtType='conv'):
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

numConvLayers = 0
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
def convLayer(x, filterShape, poolShape):
    global numConvLayers
    inputChannels = x.shape[3]
    convFilt = tf.get_variable('filt' + str(numConvLayers), \
        [filterShape[0], filterShape[1], inputChannels, filterShape[2]], \
        initializer=normInit)
    bias = tf.get_variable('bias' + str(numConvLayers), \
        [filterShape[2]], initializer=zeroInit)

    # setup layer conv, batch norm, and pooling layers.
    logit = tf.nn.conv2d(x, convFilt, strides=[1,1,1,1], padding='SAME') + bias
    normed, gamma, beta, mean, variance = \
        batchNormLayer(logit, filterShape[2], numConvLayers)
    layer = tf.nn.relu(normed)
    pooled = tf.nn.max_pool(layer, \
                ksize=[1,poolShape[0], poolShape[1], 1], \
                strides=[1,poolShape[0], poolShape[1], 1], \
                padding='SAME')


    numConvLayers += 1
    return pooled, [convFilt, bias, gamma, beta, mean, variance]


# fullConnLayer
# Define a fully connected layer with batch normalization
# @param x - the input tensor
# @param numOutputNodes - number of output nodes
#
# @return outputLayer, list of all variables
def fullConnLayer(x, numOutputNodes):
    global numConvLayers
    inputChannels = x.shape[1]

    matFilt = tf.get_variable('filt' + str(numConvLayers), \
        [inputChannels, numOutputNodes], initializer=normInit)
    bias = tf.get_variable('bias' + str(numConvLayers), \
        [numOutputNodes], initializer=zeroInit)

    logit = tf.matmul(x, matFilt) + bias
    normed, gamma, beta, mean, variance = \
        batchNormLayer(logit, numOutputNodes, numConvLayers, 'mult')
    layer = tf.nn.relu(normed)

    numConvLayers += 1
    return layer, [matFilt, bias, gamma, beta, mean, variance]


###################### Define the network.

variables = []

# define inference
layer1, tmp = convLayer(x, [3,3,32], [2,2])
print(layer1)
variables += tmp

layer2, tmp = convLayer(layer1, [5,5,64], [2,2])
variables += tmp
print(layer2)

layer3, tmp = convLayer(layer2, [3,3,128], [2,2])
variables += tmp
print(layer3)

layer4, tmp = convLayer(layer3, [7,7,64], [2,2])
variables += tmp
print(layer4)

layer5, tmp = convLayer(layer4, [5,5,128], [3,3])
variables += tmp
print(layer5)

layer6, tmp = convLayer(layer5, [7,7,32], [2,2])
variables += tmp
print(layer6)

sh = layer6.shape
flattened = tf.reshape(layer6, shape=[-1, sh[1]*sh[2]*sh[3]])
print(flattened)

#full1, tmp = fullConnLayer(flattened, 1024)
#variables += tmp
#print(full1)

full2, tmp = fullConnLayer(flattened, 1024)
variables += tmp
print(full2)

outputMat = tf.get_variable('outputMat', [full2.shape[1], 2], initializer=normInit)
outputBias = tf.get_variable('outputBias', [2], initializer=zeroInit)

outputLogit = tf.matmul(full2, outputMat) + outputBias
print(outputLogit)
variables = variables + [outputMat, outputBias]

print(variables[0])
print(variables[5])

############################ init saver and loss function

# init tensorflow saver
saver = tf.train.Saver(variables)

# define loss function
crossEntropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=outputLogit)
loss = tf.reduce_mean(crossEntropy)


########################### Define Training steps

######## define training step

learningRate = 0.005

optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
trainStep = optimizer.minimize(loss)

###### define accuracy functions

# output of equals is integers
actualClass = tf.argmax(y, axis=1)
predictedClass = tf.argmax(outputLogit, axis=1)

equals = tf.equal(actualClass, predictedClass)

# cast integers to float for reduce mean to work correctly.
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))

############################# Define batch code

file = np.load('/home/ianran/feed.npz')
#file = np.load('output.npz')
xTrain = file['x']
yTrain = file['y']

print(xTrain.dtype)

xTrain = xTrain.astype(np.float32)
yTrain = yTrain.astype(np.float32)

xValid = xTrain[0:50]
yValid = yTrain[0:50]
#xValid = xTrain
#yValid = yTrain

xTrain = xTrain[50:len(xTrain)]
yTrain = yTrain[50:len(yTrain)]

print(xTrain.dtype)

print(xTrain.shape)
print(yTrain.shape)

# this function goes through and pulls random
# parts of the training data out
# @param filenames - a list of all possible filenames to pull from
# @param batchSize - the list of batchSize
#
# @return - images, labels as numpy arrays
def pullRandomBatch(batchSize):
    # select random indcies of filenames
    indicies = np.random.choice(len(yTrain), batchSize, replace=False)
    #print(indicies)
    return xTrain[indicies], yTrain[indicies]



########################### Start session

sess = tf.Session()

# comment the restore call and uncoment the global initializer to restart the training the process.
#saver.restore(sess, 'V6')
sess.run(tf.global_variables_initializer())


######################### Define Training

# define training loop
numIterations = 2000
numToValidate = 50
numToSave = 1000000



lossSum = 0.0
for i in range(numIterations):
    if i % 5 == 0:
        print('itr: ' + str(i))
    train, label = pullRandomBatch(batchSize)

    #print(train.shape)
    #print(label.shape)
    #print(train.dtype)
    #print(label.dtype)

    #print(train[0])
    #print(label[0])

    feed = {x: train, y: label, trainPhase: True}

    lossSum += sess.run(loss, feed_dict=feed)
    sess.run(trainStep, feed_dict=feed)

    if (i+1) % numToSave == 0:
        # save the current model
        #saver.save(sess, 'layer2/layer2ShortFilters', global_step=i)
        xyz =1

    if i % numToValidate == 0:
        feed = {x: xValid, y: yValid, trainPhase: False}

        #print(xValid)
        #print(yValid)

        outputs = sess.run(outputLogit, feed_dict=feed)
        acc = sess.run(accuracy, feed_dict=feed)
        print('iteration num: ' + str(i))
        print('Validation accuracy = ' + str(acc))
        print('Avg loss = ' + str(lossSum / numToValidate))
        print('variables[0] output (3,3,32)')
        print(sess.run(variables[0]))
        print('bias')
        print(sess.run(variables[1]))
        print('outputs')
        print(outputs)
        lossSum = 0.0

saveName = 'modelV2'
print('I done, and saving model to' + saveName)
saver.save(sess, saveName)






#

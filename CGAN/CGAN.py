# CGAN.py
# Written Ian Rankin September 2018
#
# Conditional Generative Adversarial Network
# This consists of a Generative network and a discrimitive network.
# The generative network generates fake images, and the discrimitive
# network tries to discrimnate between fake and real images.
#
# Generative network:
# G(z) = x     : z - randomly sampled vector.
#
# x - images
# Discrimitive network:
# D(x) = labels
#
#
#
# Training G :
# argmax cross-entropy(v, D(G(z)))
# v : label indicating fake image
# cross-entropy loss function
#
# note the use of arg max instead of argmin to maximize the loss
#
#
# Training D :
# argmin crossEntropy(y, D(x))
# x : mixture of images from dataset, and simulated images G(z)
#
#

import numpy as np
import tensorflow as tf
import CNNUtility as cnn

# image shape for MNIST
imageShape = (30, 30, 1)
print(imageShape)

# random vector input
z = tf.placeholder(tf.float32, shape=[None, 32])
trainPhaseGen = tf.placeholder(tf.bool)
trainPhaseDis = tf.placeholder(tf.bool)

x = tf.placeholder(tf.float32, shape=[None, 30, 30, 1])
y = tf.placeholder(tf.float32, shape=[None, 2])
disInputGen = tf.placeholder(tf.bool)


genTrainableVars = []
genOtherVars = []
with tf.variable_scope('generative'):
    # fully connected networks
    fc1, trainableVars, otherVars = cnn.fullConnLayer(z, 200,trainPhaseGen)
    genTrainableVars += trainableVars
    genOtherVars += otherVars

    fc2, trainableVars, otherVars = cnn.fullConnLayer(fc1, 400,trainPhaseGen)
    genTrainableVars += trainableVars
    genOtherVars += otherVars

    ###################################### Re-shape vector to image.
    # reshaped to small image with many layers.
    reShapedImage = tf.reshape(fc2, shape=[-1, 5, 5, 16])
    print(reShapedImage)

    genImg1, trainableVars, otherVars = cnn.transposeConvLayer(reShapedImage, \
        [5,5,64], [10,10], trainPhaseGen)
    genTrainableVars += trainableVars
    genOtherVars += otherVars

    print(genImg1)

    genImg2, trainableVars, otherVars = cnn.transposeConvLayer(genImg1, \
        [5,5,32], [20,20], trainPhaseGen)
    genTrainableVars += trainableVars
    genOtherVars += otherVars

    genImg3, trainableVars, otherVars = cnn.transposeConvLayer(genImg2, \
        [5,5,32], [30,30], trainPhaseGen)
    genTrainableVars += trainableVars
    genOtherVars += otherVars

    print(genImg3)

    normInit = tf.truncated_normal_initializer(0, .05, dtype=tf.float32)
    zeroInit = tf.constant_initializer(0.0, dtype=tf.float32)

    convFilt = tf.get_variable('finalGenFilt', \
        [1, 1, genImg3.shape[3], 1], \
        initializer=normInit)
    bias = tf.get_variable('finalGenBias', \
        [1], initializer=zeroInit)

    genTrainableVars += [convFilt, bias]

    logitGen = tf.nn.conv2d(genImg3, convFilt, strides=[1,1,1,1], padding='SAME') + bias
    outputGen = tf.nn.relu(logitGen)
    print(outputGen)


################################ Discrimitive network

disInput = tf.cond(disInputGen, true_fn=outputGen, false_fn=disInputGen)

disTrainableVars = []
disOtherVars = []
with tf.variable_scope('discrimitive'):
    # convolutional
    disImg1, trainableVars, otherVars = convLayer(disInput , [3,3,64], [2,2], trainPhaseDis)
    disTrainableVars += trainableVars
    disOtherVars += otherVars

    disImg2, trainableVars, otherVars = convLayer(disImg1 , [5,5,16], [3,3], trainPhaseDis)
    disTrainableVars += trainableVars
    disOtherVars += otherVars

    ################################## Flattened discrimatve
    flattenedDis = tf.reshape(disImg2, shape=[-1, 400])

    fc1, trainableVars, otherVars = fullConnLayer(flattenedDis, 100, trainPhaseDis)
    disTrainableVars += trainableVars
    disOtherVars += otherVars
    outputDis, trainableVars, otherVars = fullConnLayer(flattenedDis, 2, trainPhaseDis)
    disTrainableVars += trainableVars
    disOtherVars += otherVars

#################### define loss and train functions functions

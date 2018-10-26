# CGANetworkMNIST.py
# Written Ian Rankin October 2018
#
# This is the architecture for the
#
#
# This consists of a Generative network and a discrimitive network.
# The generative network generates fake images, and the discrimitive
# network tries to discrimnate between fake and real images.
#
# Code based vaguly on this paper for generic GAN:
# http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
# and paper on conditional GAN:
# https://arxiv.org/pdf/1411.1784.pdf
#
#
# Generative network:
# G(z) = x     : z - randomly sampled vector.
#
# x - images
# Discrimitive network:
# D(x) = labels

import tensorflow as tf
import CNNUtility as cnn


imageShape = (28, 28, 1)


# generativeNetworkMNIST
# This creates the network for the generative side for an MNIST image
# 2 fully connected layers followed by 3 convolutional layers.
#
# @return   input random vector
#           output tensor
#           train phase variable for generative network
#           list of trainable variables
#           list of other variables
def generativeNetworkMNIST():
    # random vector input
    z = tf.placeholder(tf.float32, shape=[None, 32])
    trainPhaseGen = tf.placeholder(tf.bool)

    genTrainableVars = []
    genOtherVars = []
    with tf.variable_scope('generative'):
        # fully connected networks
        fc1, trainableVars, otherVars = cnn.fullConnLayer(z, 400,trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        fc2, trainableVars, otherVars = cnn.fullConnLayer(fc1, 784,trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        ###################################### Re-shape vector to image.
        # reshaped to small image with many layers.
        reShapedImage = tf.reshape(fc2, shape=(-1, 7, 7, 16))
        print(reShapedImage)

        genImg1, trainableVars, otherVars = cnn.transposeConvLayer(reShapedImage, \
            [7,7,64], [14,14], [1,2,2,1], trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        print('genImg1')
        print(genImg1)

        #genImg2, trainableVars, otherVars = cnn.transposeConvLayer(genImg1, \
        #    [5,5,32], [15,20], trainPhaseGen)
        #genTrainableVars += trainableVars
        #genOtherVars += otherVars

        #print('genImg2')
        #print(genImg2)

        genImg3, trainableVars, otherVars = cnn.transposeConvLayer(genImg1, \
            [5,5,32], [28,28], [1,2,2,1], trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        print('genImg3')
        print(genImg3)

        normInit = tf.truncated_normal_initializer(0, .05, dtype=tf.float32)
        zeroInit = tf.constant_initializer(0.0, dtype=tf.float32)

        convFilt = tf.get_variable('finalGenFilt', \
            [3, 3, genImg3.shape[3], 1], \
            initializer=normInit)
        bias = tf.get_variable('finalGenBias', \
            [1], initializer=zeroInit)

        genTrainableVars += [convFilt, bias]

        logitGen = tf.nn.conv2d(genImg3, convFilt, strides=[1,1,1,1], padding='SAME') + bias
        outputGen = tf.nn.sigmoid(logitGen)
        print(outputGen)

        return z, outputGen, trainPhaseGen, genTrainableVars, genOtherVars



# discrimatveNetworkMNIST
# This creates the network for the discrimatve side for an MNIST image
# 2 fully connected layers followed by 3 convolutional layers.
#
# @return   input random vector
#           output tensor
#           input train phase variable for discrimatve network
#           input select input tensor (true = input given from generative network)
#           list of trainable variables
#           list of other variables
def discrimativeNetworkMNIST(outputGen):
    x = tf.placeholder(tf.float32, shape=[None, imageShape[0], imageShape[1], imageShape[2]])
    trainPhaseDis = tf.placeholder(tf.bool)
    disInputGen = tf.placeholder(tf.bool)


    disInput = tf.cond(disInputGen, true_fn= lambda:outputGen, false_fn= lambda:x)

    disTrainableVars = []
    disOtherVars = []
    with tf.variable_scope('discrimitive'):
        # convolutional
        disImg1, trainableVars, otherVars = cnn.convLayer(disInput , [3,3,64], [2,2], trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars

        disImg2, trainableVars, otherVars = cnn.convLayer(disImg1 , [5,5,16], [2,2], trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars

        ################################## Flattened discrimatve
        flattenedDis = tf.reshape(disImg2, shape=[-1, 784])

        fc1, trainableVars, otherVars = cnn.fullConnLayer(flattenedDis, 100, trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars
        outputDis, trainableVars, otherVars = cnn.fullConnLayer(flattenedDis, 11, trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars

    return x, outputDis, trainPhaseDis, disInputGen, disTrainableVars, disOtherVars

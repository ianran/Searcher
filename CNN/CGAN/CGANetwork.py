# CGANetwork.py
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


imageShape = (405, 720, 3)


# generativeNetworkMNIST
# This creates the network for the generative side for an MNIST image
# 2 fully connected layers followed by 3 convolutional layers.
#
# @return   input random vector
#           output tensor
#           train phase variable for generative network
#           list of trainable variables
#           list of other variables
def generativeNetwork():
    # random vector input
    z = tf.placeholder(tf.float32, shape=[None, 256])
    trainPhaseGen = tf.placeholder(tf.bool)

    genTrainableVars = []
    genOtherVars = []
    with tf.variable_scope('generative'):
        # fully connected networks
        fc1, trainableVars, otherVars = cnn.fullConnLayer(z, 720,trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        fc2, trainableVars, otherVars = cnn.fullConnLayer(fc1, 1440,trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        ###################################### Re-shape vector to image.
        # reshaped to small image with many layers.
        reShapedImage = tf.reshape(fc2, shape=(-1, 5, 9, 32))
        print('GenerativeNetwork network')
        print(reShapedImage)

        genImg1, trainableVars, otherVars = cnn.transposeConvLayer(reShapedImage, \
            [11,11,128], [15,27], [1,3,3,1], trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        print(genImg1)
        # (15, 27)
        genImg2, trainableVars, otherVars = cnn.transposeConvLayer(genImg1, \
            [11,11,128], [45,80], [1,3,3,1], trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        print(genImg2)
        # (45, 80)
        genImg3, trainableVars, otherVars = cnn.transposeConvLayer(genImg2, \
            [11,11,128], [135,240], [1,3,3,1], trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        print(genImg3)
        # (135, 240)
        genImg4, trainableVars, otherVars = cnn.transposeConvLayer(genImg3, \
            [11,11,128], [405,720], [1,3,3,1], trainPhaseGen)
        genTrainableVars += trainableVars
        genOtherVars += otherVars

        # (405, 720)
        print(genImg4)


        normInit = tf.truncated_normal_initializer(0, .05, dtype=tf.float32)
        zeroInit = tf.constant_initializer(0.0, dtype=tf.float32)

        convFilt = tf.get_variable('finalGenFilt', \
            [3, 3, genImg3.shape[3], 3], \
            initializer=normInit)
        bias = tf.get_variable('finalGenBias', \
            [3], initializer=zeroInit)

        genTrainableVars += [convFilt, bias]

        logitGen = tf.nn.conv2d(genImg4, convFilt, strides=[1,1,1,1], padding='SAME') + bias
        outputGen = tf.nn.sigmoid(logitGen)
        print(outputGen)

        return z, outputGen, trainPhaseGen, genTrainableVars, genOtherVars



# discrimatveNetwork
# This creates the network for the discrimatve side for an MNIST image
# 2 fully connected layers followed by 3 convolutional layers.
#
# @return   input random vector
#           output tensor
#           input train phase variable for discrimatve network
#           input select input tensor (true = input given from generative network)
#           list of trainable variables
#           list of other variables
def discrimativeNetwork(outputGen):
    x = tf.placeholder(tf.float32, shape=[None, imageShape[0], imageShape[1], imageShape[2]])
    trainPhaseDis = tf.placeholder(tf.bool)
    disInputGen = tf.placeholder(tf.bool)


    disInput = tf.cond(disInputGen, true_fn= lambda:outputGen, false_fn= lambda:x)

    disTrainableVars = []
    disOtherVars = []
    with tf.variable_scope('discrimitive'):
        # convolutional
        # (405,720)
        print('Discrimitive network')
        print(disInput)
        disImg1, trainableVars, otherVars = cnn.convLayer(disInput, [5,5,64], [3,3], trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars

        print(disImg1)

        # (135,240)
        disImg2, trainableVars, otherVars = cnn.convLayer(disImg1, [5,5,64], [3,3], trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars

        print(disImg2)

        # (45, 80)
        disImg3, trainableVars, otherVars = cnn.convLayer(disImg2, [7,7,128], [3,3], trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars

        print(disImg3)

        # (15, 27)
        disImg4, trainableVars, otherVars = cnn.convLayer(disImg3, [5,5,32], [3,3], trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars

        print(disImg4)

        # (5, 9)
        #disImg5, trainableVars, otherVars = cnn.convLayer(disImg4, [5,5,32], [3,3], trainPhaseDis)
        #disTrainableVars += trainableVars
        #disOtherVars += otherVars

        # (1440)
        #print(disImg5)

        ################################## Flattened discrimatve
        flattenedDis = tf.reshape(disImg4, shape=[-1, 1440])

        print(flattenedDis)

        fc1, trainableVars, otherVars = cnn.fullConnLayer(flattenedDis, 720, trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars
        outputDis, trainableVars, otherVars = cnn.fullConnLayer(flattenedDis, 3, trainPhaseDis)
        disTrainableVars += trainableVars
        disOtherVars += otherVars

    return x, outputDis, trainPhaseDis, disInputGen, disTrainableVars, disOtherVars

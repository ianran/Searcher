# CGANRealData.py
# Written Ian Rankin September 2018
#
# Conditional Generative Adversarial Network for the real data.
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
import CGANetwork as cgan
import Dataset as dt
#import scipy.misc as mis
#import imageio

# image shape for MNIST
imageShape = (405, 720, 3)
numOutputClasses = 3
randomVecSize = 256
print(imageShape)

z, outputGen, trainPhaseGen, genTrainableVars, \
    genOtherVars = cgan.generativeNetwork()

x, outputDis, trainPhaseDis, disInputGen, disTrainableVars, \
    disOtherVars = cgan.discrimativeNetwork(outputGen)

y = tf.placeholder(tf.float32, shape=[None, numOutputClasses])




####################### define accuracy, and encode functions
actualClass = tf.argmax(y, axis=1)
predictedClass = tf.argmax(outputDis, axis=1)
equals = tf.equal(actualClass, predictedClass)

# cast integers to float for reduce mean to work correctly.
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))

####################### define loss and train functions functions

crossEntropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=outputDis)
print(crossEntropy)
loss = tf.reduce_mean(crossEntropy)
print(loss)

saver = tf.train.Saver(genTrainableVars + genOtherVars + disTrainableVars + disOtherVars)

# discrimitive optimizer
discrimatveOptimizer = tf.train.AdamOptimizer(learning_rate=0.001)
discTrainStep = discrimatveOptimizer.minimize(loss, var_list=disTrainableVars)

# generative optimizer is the negative of the loss to maximze the loss
generativeOptimizer = tf.train.AdamOptimizer(learning_rate=0.01)
genTrainStep = generativeOptimizer.minimize(-loss, var_list=genTrainableVars)


########################### Read in training data.

# read in data
trainImagesFull, trainLabelsFull, validImages, validLabelsFull \
        = dt.readData()


########################## Training

sess = tf.Session()

sess.run(tf.global_variables_initializer())

numEpochs = 10000
numDisc = 5
numBatch = 25

allSynthLabels = np.zeros((numBatch, numOutputClasses), np.float32)
allSynthLabels[:,numOutputClasses-1] = 1.0

# Validation network
# realImages -> discrimative network -> accuracy(with real labels)
#
validFeed = {trainPhaseGen: False, trainPhaseDis: False, disInputGen: False}
#validImages = np.resize(data.validation.images, [5000, imageShape[0], imageShape[1], imageShape[2]])
print(validImages.shape)
validFeed[x] = validImages
#validLabels = np.zeros((validLabelsFull.shape[0], numOutputClasses))
#for i in range(validLabelsFull.shape[0]):
#    validLabels[i][validLabelsFull[i]] = 1.0

validFeed[y] = validLabelsFull
validFeed[z] = np.zeros((validImages.shape[0],randomVecSize))

feedGenTrain = {trainPhaseGen: True, trainPhaseDis: False, disInputGen: True}
feedGenTrain[y] = allSynthLabels
feedGenTrain[x] = np.zeros((numBatch, imageShape[0], imageShape[1], imageShape[2]))



for i in range(numEpochs):
    print('epoch = ' + str(i))
    ################### Train discrimative network
    feedGen = {trainPhaseGen: False, trainPhaseDis: False, disInputGen: True}
    for j in range(numDisc):
        # generate synthesized images, and labels
        #
        # randomVector -> GenerativeNetwork -> synth images
        #
        randVec = np.random.normal(0.0,1.0,(numBatch//2,randomVecSize))
        feedGen[z] = randVec
        synthImages = sess.run(outputGen, feed_dict=feedGen)
        synthLabels = np.zeros((numBatch//2,numOutputClasses), np.float32)
        synthLabels[:,numOutputClasses-1] = 1.0

        if (j == 0 and i % 5 == 0):
            # save a synth image a few times.
            #write_jpeg('/scratch/ianran/img/synthImage'+str(i)+'.jpg', synthImages[0])
            dt.write_jpeg('img/synthImage'+str(i)+'.jpg', synthImages[0], imageShape)


        # append synth images with real images.
        realImages, realLabels = dt.getNextBatch(trainImagesFull, trainLabelsFull, numBatch//2)

        #print('real images')
        #print(realImages.shape)
        #print('synth images')
        #print(synthImages.shape)
        #print('real labels')
        #print(realLabels.shape)
        #print('synth labels')
        #print(synthLabels.shape)

        trainImages = np.append(synthImages, realImages, 0)
        trainLabels = np.append(synthLabels, realLabels, 0)

        #print(trainImages.shape)

        ######### set the training phases and input to the discrimater.
        #
        # synthImages + realImages -> discrimativeNetwork -> minimize loss
        #
        feed = {trainPhaseGen: False, trainPhaseDis: True, disInputGen: False}
        feed[x] = trainImages
        feed[y] = trainLabels
        feed[z] = np.zeros((numBatch,randomVecSize))

        #print('About to train Discrimitive network')
        sess.run(discTrainStep, feed_dict=feed)
        #print('Trainged disc network')

    ################# train generative network
    # set training phase to generative network, and the input of discrimative network
    # to be the generative network
    #
    # random vector -> generativeNetwork -> discrimativeNetwork -> maxmize loss
    #
    randVec = np.random.normal(0,1.0,(numBatch,randomVecSize))
    feedGenTrain[z] = randVec

    # train generative network
    sess.run(genTrainStep, feed_dict=feedGenTrain)

    ######### validate network
    if (i % 50 == 0 or i == (numEpochs - 1)):
        acc = sess.run(accuracy, feed_dict=validFeed)
        print('Validation accuracy = ' + str(acc))


####################### After training.
saver.save(sess, '../../models/cgan4', global_step=numEpochs)


validFeed[x] = validImages
validFeed[y] = validLabels
acc = sess.run(accuracy, feed_dict=validFeed)

print('test accuracy = ')
print(acc)

genTestFeed = {trainPhaseGen: False, trainPhaseDis: False, disInputGen: True}
randVec = np.random.normal(0.0,1.0,(100,randomVecSize))
genTestFeed[z] = randVec
synthImages = sess.run(outputGen, feed_dict=genTestFeed)

for i in range(synthImages.shape[0]):
    print('bob')
    #write_jpeg('/scratch/ianran/img/testSynth'+str(i)+'.jpg', synthImages[i])





#

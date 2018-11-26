# TradCNN.py
# Written Ian Rankin September 2018
#
# This is just a traditonal CNN

import numpy as np
import tensorflow as tf
import CNNUtility as cnn
import CGANetwork as cgan
import Dataset as dt
#import scipy.misc as mis
#import imageio

# image shape for MNIST
imageShape = (405, 720, 3)
numOutputClasses = 2
randomVecSize = 256
print(imageShape)

x = tf.placeholder(tf.float32, shape=[None, imageShape[0], imageShape[1], imageShape[2]])
output, trainPhase, trainableVars, otherVars = cgan.CNN_Network(x, numOutputClasses)

y = tf.placeholder(tf.float32, shape=[None, numOutputClasses])

jpegOpGen = dt.jpegGraph(x[0])


####################### define accuracy, and encode functions
actualClass = tf.argmax(y, axis=1)
predictedClass = tf.argmax(output, axis=1)
equals = tf.equal(actualClass, predictedClass)

# cast integers to float for reduce mean to work correctly.
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))
correct = tf.reduce_sum(tf.cast(equals, tf.float32))

####################### define loss and train functions functions

crossEntropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output)
print(crossEntropy)
loss = tf.reduce_mean(crossEntropy)
print(loss)

saver = tf.train.Saver(trainableVars + otherVars)

# discrimitive optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
trainStep = optimizer.minimize(loss, var_list=trainableVars)


########################### Read in training data.

# read in data
#trainImagesFull, trainLabelsFull, validImagesFull, validLabelsFull \
#      = dt.readData()
trainImagesPeople, trainImagesNoPeople, validImages, validLabels \
    = dt.readDataNormalized()


########################## Training

sess = tf.Session()

sess.run(tf.global_variables_initializer())

numEpochs = 50
numDisc = 5
numBatch = 35


# Validation network
# realImages -> discrimative network -> accuracy(with real labels)
# @param labels - all of the labels to validate using
# @param images - all of the images to validate using
#
# @return - accuracy of dataset.
def validate(labels, images, batchSize, sess):
   validFeed = {trainPhase: False}
   numValidBatches = len(labels) // batchSize
   extraData = len(labels) % batchSize

   correctlyIdent = 0
   for j in range(numValidBatches):
      validFeed[x] = images[j*numBatch:(j+1)*numBatch]
      validFeed[y] = labels[j*numBatch:(j+1)*numBatch]

      correctlyIdent += sess.run(correct, feed_dict=validFeed)

   # extra data from max number of batches to finish off validation check
   if (extraData > 0):
      validFeed[x] = images[numValidBatches*numBatch:len(labels)]
      validFeed[y] = labels[numValidBatches*numBatch:len(labels)]

      correctlyIdent += sess.run(correct, feed_dict=validFeed)

   return correctlyIdent / len(labels)


# test network
# confusion matrix
# realImages -> discrimative network -> accuracy(with real labels)
# @param labels - all of the labels to validate using
# @param images - all of the images to validate using
#
# @return - accuracy of dataset.
def testNetwork(labels, images, batchSize, sess):
   validFeed = {trainPhase: False}
   numValidBatches = len(labels) // batchSize
   extraData = len(labels) % batchSize

   print('testNetwork called')
   predictedLabels = np.empty(len(labels), dtype=np.int32)
   for j in range(numValidBatches):
      validFeed[x] = images[j*numBatch:(j+1)*numBatch]
      validFeed[y] = labels[j*numBatch:(j+1)*numBatch]

      predictedLabels[j*numBatch:(j+1)*numBatch] = sess.run(predictedClass, feed_dict=validFeed)

   # extra data from max number of batches to finish off validation check
   if (extraData > 0):
      validFeed[x] = images[numValidBatches*numBatch:len(labels)]
      validFeed[y] = labels[numValidBatches*numBatch:len(labels)]

      predictedLabels[numValidBatches*numBatch:len(labels)] = sess.run(predictedClass, feed_dict=validFeed)

   print('before casting into argmax')
   labelsClass = np.empty(len(labels), dtype=np.int32)
   labelsClass = np.argmax(labels, axis=1)

   #preClass = np.empty(len(labels), dtype=np.int32)
   #preClass = np.argmax(predictedLabels, axis=1)
   preClass = predictedLabels

   print(labelsClass.shape)
   print(preClass.shape)

   # create confusion matrix
   confMat = np.zeros((labels.shape[1],labels.shape[1]))
   for i in range(len(labels)):
       confMat[preClass[i],labelsClass[i]] += 1
   acc = (confMat[0,0] + confMat[1,1]) / (confMat[0,0] + confMat[0,1] + confMat[1,0] + confMat[1,1])
   print('0 = noPeople, 1 = people')
   print('x : predicted class, y : actual class')
   print(confMat)
   print('accuracy = ' + str(acc))



###################### Training

for i in range(numEpochs):
   numBatchesPerEpoch, epochTuple = dt.generateEpoch(trainImagesPeople, \
        trainImagesNoPeople, numBatch)
   k = 0
   print('epoch = ' + str(i) + ' with ' + str(numBatchesPerEpoch) + ' batches')
   for j in range(numBatchesPerEpoch):
       feed = {trainPhase: True}
       print('\tepoch batch = ' + str(j))
       images, labels, k = dt.getNextBatchEpoch(k, epochTuple, numBatch)
       print('k = ' + str(k))
       feed[x] = images
       feed[y] = labels

       sess.run(trainStep, feed_dict=feed)

   ######### validate network and save model
   if (i % 3 == 0 or i == (numEpochs - 1)):
      #print('Validation accuracy = ' + \
      #    str(validate(validLabelsFull, validImagesFull, numBatch, sess)))
      testNetwork(validLabelsFull, validImagesFull, numBatch, sess)
   if i % 6 == 99 or i == (numEpochs - 1):
       saver.save(sess, '../../models/cnn5', global_step=i)

####################### After training.


acc = validate(validLabelsFull, validImagesFull, numBatch, sess)

print('test accuracy = ')
print(acc)




#

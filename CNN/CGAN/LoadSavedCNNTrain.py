# TradCNN.py
# Written Ian Rankin September 2018
#
# Load saved CNN model, and retrain for a while.

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
print(imageShape)

x = tf.placeholder(tf.float32, shape=[None, imageShape[0], imageShape[1], imageShape[2]])
#output, trainPhase, trainableVars, otherVars = cgan.CNN_Network(x, numOutputClasses)
output, trainPhase, trainableVars, otherVars = cgan.CNN_NetworkLarge(x, numOutputClasses)

y = tf.placeholder(tf.float32, shape=[None, numOutputClasses])

jpegOp = dt.jpegGraph(x[0])


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
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
trainStep = optimizer.minimize(loss, var_list=trainableVars)


########################### Read in training data.

# read in data
#trainImagesFull, trainLabelsFull, validImagesFull, validLabelsFull \
#      = dt.readData()



########################## Training

sess = tf.Session()

sess.run(tf.global_variables_initializer())
# Load in tensor flow model from previous training
saver.restore(sess, '../../models/cnn10-99')

print('Starting reading in dataset')
trainImagesPeople, trainImagesNoPeople, validImagesFull, validLabelsFull \
    = dt.readDataNormalized()
print('Completed reading in dataset')

numEpochs = 100
numBatch = 40



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
feed = {trainPhase: True}

# get next batch of images
batchImages = np.empty((numBatch,imageShape[0],imageShape[1],imageShape[2]), np.float32)
batchLabels = np.zeros((numBatch, 2), np.float32)

for i in range(numEpochs):
   numBatchesPerEpoch, epochTuple = dt.generateEpoch(len(trainImagesPeople), \
        len(trainImagesNoPeople), numBatch)
   print('epoch indcies = ')
   print(epochTuple[0])
   print('epochSelector = ')
   print(epochTuple[1])

   k = 0
   lossTotal = 0.0
   print('EPOCH = ' + str(i) + ' with ' + str(numBatchesPerEpoch) + ' batches')
   for j in range(numBatchesPerEpoch):
       print('epoch batch = ' + str(j))
       k = dt.getNextBatchEpoch(k, epochTuple, numBatch, \
            trainImagesPeople, trainImagesNoPeople, batchImages, batchLabels)
       #print('k = ' + str(k))
       feed[x] = batchImages
       feed[y] = batchLabels

       tmp, lossCur, outputVec, crossVec = sess.run([trainStep, loss, output, crossEntropy], feed_dict=feed)
       lossTotal += lossCur
       #print('Loss current = ' + str(lossCur))
       #print('OutputVector = ')
       #print(outputVec)
       #print('cross vector = ')
       #print(crossVec)
       if j % 30 == 0:
           dt.write_jpeg('/scratch/ianran/img2/' + str(i) + '-' + str(j) + '.jpg', \
                batchImages[0],imageShape)
           print('This set of batch labels')
           print(batchLabels)
           #dt.writeJPEGGivenGraph('/scratch/ianran/img2/' + str(i) + '-' + str(j) + '.jpg',
            #    sess, jpegOp)

   ######### validate network and save model
   print('Epoch total loss = ' + str(lossTotal))
   if (i % 3 == 0 or i == (numEpochs - 1)):
      #print('Validation accuracy = ' + \
      #    str(validate(validLabelsFull, validImagesFull, numBatch, sess)))
      print('VALIDATION')
      testNetwork(validLabelsFull, validImagesFull, numBatch, sess)
      print('TEST SET on last batch')
      print('batchLabels')
      print(batchLabels)
      testNetwork(batchLabels, batchImages, numBatch, sess)

   if i % 6 == 5 or i == (numEpochs - 1):
       saver.save(sess, '../../models/cnn11', global_step=i)

####################### After training.


acc = validate(validLabelsFull, validImagesFull, numBatch, sess)

print('test accuracy = ')
print(acc)




#

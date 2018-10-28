TradCNN.py
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
numOutputClasses = 3
randomVecSize = 256
print(imageShape)

x = tf.placeholder(tf.float32, shape=[None, imageShape[0], imageShape[1], imageShape[2]])
output, trainPhase, trainableVars, otherVars = cgan.CNN_Network(x)

y = tf.placeholder(tf.float32, shape=[None, numOutputClasses])

jpegOpGen = dt.jpegGraph(x)


####################### define accuracy, and encode functions
actualClass = tf.argmax(y, axis=1)
predictedClass = tf.argmax(outputDis, axis=1)
equals = tf.equal(actualClass, predictedClass)

# cast integers to float for reduce mean to work correctly.
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))
correct = tf.reduce_sum(tf.cast(equals, tf.float32))

####################### define loss and train functions functions

crossEntropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=outputDis)
print(crossEntropy)
loss = tf.reduce_mean(crossEntropy)
print(loss)

saver = tf.train.Saver(trainableVars + otherVars)

# discrimitive optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
trainStep = discrimatveOptimizer.minimize(loss, var_list=trainableVars)


########################### Read in training data.

# read in data
trainImagesFull, trainLabelsFull, validImagesFull, validLabelsFull \
      = dt.readData()


########################## Training

sess = tf.Session()

sess.run(tf.global_variables_initializer())

numEpochs = 4000
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

      correctlyIdent += sess.run(accuracy, feed_dict=validFeed)

   # extra data from max number of batches to finish off validation check
   if (extraData > 0):
      validFeed[x] = images[numValidBatches*numBatch:len(validLabelsFull)]
      validFeed[y] = labels[j*numBatch:(j+1)*numBatch]

      correctlyIdent += sess.run(accuracy, feed_dict=validFeed)

   return correctlyIdent / len(labels)


###################### Training

for i in range(numEpochs):
   print('epoch = ' + str(i))
   ################### Train discrimative network
   feedDis = {trainPhaseGen: False, trainPhaseDis: False, disInputGen: True}
   for j in range(numDisc):
       sess.run(discTrainStep, feed_dict=feed)

      # generate synthesized images, and labels
      #
      # randomVector -> GenerativeNetwork -> synth images
      #
      randVec = np.random.uniform(0.0,1.0,(numBatch//2,randomVecSize))
      feedGen[z] = randVec
      synthImages = sess.run(outputGen, feed_dict=feedGen)
      synthLabels = np.zeros((numBatch//2,numOutputClasses), np.float32)
      synthLabels[:,numOutputClasses-1] = 1.0

      if (j == 0 and i % 10 == 0):
           # save a synth image a few times.
           #write_jpeg('/scratch/ianran/img/synthImage'+str(i)+'.jpg', synthImages[0])
           #dt.write_jpeg('img/synthImage'+str(i)+'.jpg', synthImages[0], imageShape)
           dt.writeJPEGGivenGraph('img/synthImage'+str(i)+'.jpg', sess, jpegOpGen)

      # append synth images with real images.
      realImages, realLabels = dt.getNextBatch(trainImagesFull, trainLabelsFull, numBatch//2)
      trainImages = np.append(synthImages, realImages, 0)
      trainLabels = np.append(synthLabels, realLabels, 0)

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
   randVec = np.random.uniform(0,1.0,(numBatch,randomVecSize))
   feedGenTrain[z] = randVec

   # train generative network
   sess.run(genTrainStep, feed_dict=feedGenTrain)

   ######### validate network and save model
   if (i % 50 == 49 or i == (numEpochs - 1)):
      print('Validation accuracy = ' + \
           str(validate(validLabelsFull, validImagesFull, numBatch, sess)))
      saver.save(sess, '../../models/cgan4', global_step=i)

####################### After training.




acc = validate(validLabelsFull, validImagesFull, numBatch, sess)

print('test accuracy = ')
print(acc)

print('generating synthesized images')
genTestFeed = {trainPhaseGen: False, trainPhaseDis: False, disInputGen: True}
randVec = np.random.uniform(0.0,1.0,(100,randomVecSize))
genTestFeed[z] = randVec
synthImages = sess.run(outputGen, feed_dict=genTestFeed)

for i in range(synthImages.shape[0]):
   #print('bob')
   write_jpeg('/scratch/ianran/img/testSynthReal'+str(i)+'.jpg', synthImages[i])





#

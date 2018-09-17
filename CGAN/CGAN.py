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
#import scipy.misc as mis
#import imageio

# image shape for MNIST
imageShape = (28, 28, 1)
print(imageShape)

# random vector input
z = tf.placeholder(tf.float32, shape=[None, 32])
trainPhaseGen = tf.placeholder(tf.bool)
trainPhaseDis = tf.placeholder(tf.bool)

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 11])
disInputGen = tf.placeholder(tf.bool)


################################## Generative network
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


################################ Discrimitive network

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

####################### define accuracy, and encode functions
actualClass = tf.argmax(y, axis=1)
predictedClass = tf.argmax(outputDis, axis=1)
equals = tf.equal(actualClass, predictedClass)

# cast integers to float for reduce mean to work correctly.
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))

#slice = outputGen[0,:,:,:]
#print(slice)
#scaledImage = tf.cast(slice * tf.constant(255.0, dtype=tf.float32), dtype=tf.uint8)
#print(scaledImage)
#encoder = tf.image.encode_jpeg(scaledImage)
#print(encoder)

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


########################### define batch functions

# read in the mnist data set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=False)

print(data.train.images.shape)
print(data.train.labels.shape)
print(data.train.labels[20])
print(data.train.labels[21])
print(data.train.labels[22])
print(data.train.labels[23])
print(data.train.labels[24])
print(data.train.labels[25])

def getNextBatch(batchSize):
    miniBatchImg, miniBatchLabels = data.train.next_batch(batchSize)

    resizedImages = np.resize(miniBatchImg, (batchSize,28,28,1))

    # create one hot vector
    oneHot = np.zeros((batchSize, 11), np.float32)
    for i in range(batchSize):
        oneHot[i][miniBatchLabels[i]] = 1.0

    return resizedImages, oneHot


########################## Training

sess = tf.Session()

sess.run(tf.global_variables_initializer())

numEpochs = 10000
numDisc = 5
numBatch = 50

allSynthLabels = np.zeros((numBatch,11), np.float32)
allSynthLabels[:,10] = 1.0

# Validation network
# realImages -> discrimative network -> accuracy(with real labels)
#
validFeed = {trainPhaseGen: False, trainPhaseDis: False, disInputGen: False}
print(data.validation.images.shape)
validImages = np.resize(data.validation.images, [5000,28,28,1])
print(validImages.shape)
validFeed[x] = validImages
validLabels = np.zeros((data.validation.labels.shape[0], 11))
for i in range(data.validation.labels.shape[0]):
    validLabels[i][data.validation.labels[i]] = 1.0
validFeed[y] = validLabels
validFeed[z] = np.zeros((5000,32))

feedGenTrain = {trainPhaseGen: True, trainPhaseDis: False, disInputGen: True}
feedGenTrain[y] = allSynthLabels
feedGenTrain[x] = np.zeros((50,28,28,1))

###### function to save images given 1,30,30,1 shape
#def saveImage(file, img):
#    validImage = np.resize(img, (28,28))
#    mis.imsave(file, validImage)

def write_jpeg(filepath, data):
    g = tf.Graph()
    with g.as_default():
        data_t = tf.placeholder(tf.float32, shape=[28,28,1])
        scaledImage = tf.cast(data_t * tf.constant(255.0, dtype=tf.float32), dtype=tf.uint8)
        op = tf.image.encode_jpeg(scaledImage, format='grayscale', quality=100)
        init = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        data_np = sess.run(op, feed_dict={ data_t: data })
    print(type(data_np))
    with open(filepath, 'wb') as fd:
        fd.write(data_np)


#write_jpeg('/scratch/ianran/valid0tf.jpg', validImages[0])
#saveImage('scratch/ianran/img/valid1.jpg', validImages[1])
#saveImage('scratch/ianran/img/valid2.jpg', validImages[2])
#saveImage('scratch/ianran/img/valid55.jpg', validImages[55])
#saveImage('scratch/ianran/img/valid600.jpg', validImages[600])

for i in range(numEpochs):
    print('epoch = ' + str(i))
    ########## Train discrimative network
    feedGen = {trainPhaseGen: False, trainPhaseDis: False, disInputGen: True}
    for j in range(numDisc):
        # generate synthesized images, and labels
        #
        # randomVector -> GenerativeNetwork -> synth images
        #
        randVec = np.random.normal(0.0,1.0,(numBatch//2,32))
        feedGen[z] = randVec
        synthImages = sess.run(outputGen, feed_dict=feedGen)
        synthLabels = np.zeros((numBatch//2,11), np.float32)
        synthLabels[:,10] = 1.0

        if (j == 0 and i % 5 == 0):
            # save a synth image a few times.
            #write_jpeg('/scratch/ianran/img/synthImage'+str(i)+'.jpg', synthImages[0])
            write_jpeg('img/synthImage'+str(i)+'.jpg', synthImages[0])


        # append synth images with real images.
        realImages, realLabels = getNextBatch(numBatch//2)

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
        feed[z] = np.zeros((numBatch,32))

        #print('About to train Discrimitive network')
        sess.run(discTrainStep, feed_dict=feed)
        #print('Trainged disc network')

    ######### train generative network
    # set training phase to generative network, and the input of discrimative network
    # to be the generative network
    #
    # random vector -> generativeNetwork -> discrimativeNetwork -> maxmize loss
    #
    randVec = np.random.normal(0,1.0,(numBatch,32))
    feedGenTrain[z] = randVec

    # train generative network
    sess.run(genTrainStep, feed_dict=feedGenTrain)

    ######### validate network
    if (i % 50 == 0 or i == (numEpochs - 1)):
        acc = sess.run(accuracy, feed_dict=validFeed)
        print('Validation accuracy = ' + str(acc))

saver.save(sess, '../../models/cgan2', global_step=numEpochs)

testImages = np.resize(data.test.images, (-1,28,28,1))
testLabels = np.zeros((testImages.shape[0], 11))
for i in range(testImages.shape[0]):
    testLabels[i][data.test.labels[i]] = 1.0
validFeed[x] = testImages
validFeed[y] = testLabels
acc = sess.run(accuracy, feed_dict=validFeed)

print('test accuracy = ')
print(acc)

genTestFeed = {trainPhaseGen: False, trainPhaseDis: False, disInputGen: True}
randVec = np.random.normal(0.0,1.0,(100,32))
genTestFeed[z] = randVec
synthImages = sess.run(outputGen, feed_dict=genTestFeed)

for i in range(synthImages.shape[0]):
    write_jpeg('/scratch/ianran/img/testSynth'+str(i)+'.jpg', synthImages[i])





#

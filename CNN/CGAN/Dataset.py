# Dataset.py
# Written Ian Rankin October 2018
#
# Read in the dataset

import tensorflow as tf
import numpy as np



####################### Other utility

# jpegGraph
# write out a the graph operator to write a jpeg in tensorflow.
# must be passed an image tensor with 3 channels, and must be
# scaled between 0-255
# @param image - the image tensor
#
# @returns the operation object for creating the jpeg.
def jpegGraph(image):
    scaledImage = tf.cast(image * tf.constant(255.0, dtype=tf.float32), dtype=tf.uint8)
    op = tf.image.encode_jpeg(scaledImage, format='rgb', quality=100)

# write the jpeg given the graph operation and session given.
def writeJPEGGivenGraph(filepath, sess, jpegOp):
    encodedImage = sess.run(jpegOp)

    with open(filepath, 'wb') as fd:
        fd.write(encodedImage)

# write_jpeg
# Write a jpeg using tensorflow, the code should work on joker.
# @param filepath - filepath of the image to write to.
# @param data - the image data used to to try ot write using (numpy array)
# @param imageShape - the shape of the image to be outputed.
def write_jpeg(filepath, data, imageShape):
    g = tf.Graph()
    with g.as_default():
        data_t = tf.placeholder(tf.float32, shape=[imageShape[0], imageShape[1], imageShape[2]])
        scaledImage = tf.cast(data_t * tf.constant(255.0, dtype=tf.float32), dtype=tf.uint8)
        op = tf.image.encode_jpeg(scaledImage, format='rgb', quality=100)
        init = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        data_np = sess.run(op, feed_dict={ data_t: data })
    print(type(data_np))
    with open(filepath, 'wb') as fd:
        fd.write(data_np)



########################### Read in training data.

# readData
# this function will read in the required data, and output
# numpy arrays of training data, and validation data.
#
# return    trainImages
#           trainLabels
#           validImages
#           validLabels
def readData():
    fileTrain = np.load('train.npz')
    fileValid = np.load('valid.npz')

    trainImages = fileTrain['x']
    trainLabels = fileTrain['y']

    validImages = fileValid['x']
    validLabels = fileValid['y']

    print(trainImages.shape)
    print(validImages.shape)

    #write_jpeg('test.jpg', trainImages[0], (405,720,3))
    #write_jpeg('test14.jpg', trainImages[14], (405,720,3))
    #write_jpeg('test16.jpg', trainImages[16], (405,720,3))
    #print(trainLabels)
    return trainImages, trainLabels, validImages, validLabels



########################### define batch functions

def getNextBatch(images, labels, batchSize):
    numOutputClasses = 3
    indicies = np.random.choice(len(images), batchSize, replace=False)

    miniBatchLabels = labels[indicies]



    # create one hot vector
    #oneHot = np.zeros((batchSize, numOutputClasses), np.float32)
    #for i in range(batchSize):
    #    oneHot[i][miniBatchLabels[i]] = 1.0

    return images[indicies], miniBatchLabels


# Test code
#readData()

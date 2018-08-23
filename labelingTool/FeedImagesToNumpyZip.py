# FeedImagesToNumpyZip.py
# Written Ian Rankin August 2018
#
# Designed to take as input the feed images and create a numpy zip file.
#
# Example to read the code in:
# import numpy as np
#
# file = np.load('output.npz')
#
# The images are stored in x with the shape (number images, height, width, channels)
# x = file['x']
# labels are stored in y with the shape (number of images,)
# y = file['y']
#
#

import sys
# check before importing modules if there is the corect arguments being pased.
if len(sys.argv) != 3:
    print('USAGE python FeedImagesToNumpyZip [feedDir] [labels csv]')
    sys.exit()

import numpy as np
import matplotlib.pyplot as im
import labelReader

import glob
import os



# Set the feed directory to read all images in from
feedDir = sys.argv[1]

# read in labels csv file
labels = labelReader.readLabelsDict(sys.argv[2])

# read in all filenames of images
filenames = glob.glob(feedDir + '*.jpg')

# Check if there are any images in the feed directory
if len(filenames) <= 0:
    print('Error feed directory is empty...')
    sys.exit()

testImage = im.imread(filenames[0])
s = testImage.shape
#print(s)
x = np.empty((len(filenames), s[0], s[1], s[2]))
y = np.empty(len(filenames))

for i in range(len(filenames)):

    image = im.imread(filenames[i])
    ##### If any preprocessing is desired, here is a place to do it!
    x[i] = image

    filename = os.path.basename(filenames[i])
    label = labels[filename]
    if label == 'noPeople':
        y[i] = 0
    elif label == 'people':
        y[i] = 1
    else:
        print('Error, the image: '+ filename + 'is not in the labels')
        sys.exit()

#print(x.shape)
np.savez('output.npz', x = x, y = y)

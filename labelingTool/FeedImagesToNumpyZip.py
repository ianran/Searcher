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
print('Started running', flush=True)

import sys
# check before importing modules if there is the corect arguments being pased.
if len(sys.argv) != 3 and len(sys.argv) != 4:
    print('USAGE python FeedImagesToNumpyZip [feedDir] [labels csv] opt[whiten]')
    sys.exit()

print('options good')

import numpy as np
#import matplotlib.pyplot as im
import scipy.ndimage as im
import scipy.stats as stat
import skimage.transform as misc
import labelReader

import glob
import os


print('Just work dangit')

# Set the feed directory to read all images in from
feedDir = sys.argv[1]

print('work dangit again')

# read in labels csv file
labels = labelReader.readLabelsDict(sys.argv[2])

# read in all filenames of images
filenames = glob.glob(feedDir + '*.jpg')

# Check if there are any images in the feed directory
if len(filenames) <= 0:
    print('Error feed directory is empty...')
    sys.exit()

filenames = filenames[0:50]
#image = im.imread(filenames[0])
#s = image.shape
s = (405, 720, 3)
print(s)
x = np.empty((len(filenames), s[0], s[1], s[2]))
y = np.zeros((len(filenames), 2))
#print(len(filenames))
s = (405, 720, 3)

for i in range(len(filenames)):
#for i in range(5):
    image = im.imread(filenames[i])
    #print(image)
    if (image.shape != s):
        image = misc.resize(image, s)

    #print('yo')
    #print(image)

    if len(sys.argv) == 4:
        image = stat.zscore(image)

    #print('whitened')
    #print(image)
    ##### If any preprocessing is desired, here is a place to do it!
    x[i] = image
    if i % 20 == 0:
        print(i, flush=True)
    filename = os.path.basename(filenames[i])
    label = labels[filename]
    if label == 'noPeople':
        y[i][0] = 1.0
    elif label == 'people':
        y[i][1] = 1.0
    else:
        print('Error, the image: '+ filename + 'is not in the labels')
        sys.exit()


#s = input('asd')
print(x.shape)
print(y.shape, flush=True)

#print(x)
#print(y)

np.savez('output.npz', x = x, y = y)

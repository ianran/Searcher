# Note needs getch python package
# https://pypi.python.org/pypi/getch#downloads
#
# USAGE
# place image files in same folder, must also have folders named
# people/ and noPeople/

import numpy as np
import scipy.ndimage as im
import matplotlib.pyplot as plt
import getch
import glob
import os
import sys

# the directory to read images from
feedDir = None
noPeopleDir = None
peopleDir = None

lastFile = None
lastAnswer = None

def labelSingleImage(file):
    global lastFile
    global lastAnswer
    print('f - people, j - no people, b - back')

    file = os.path.basename(file)

    image = im.imread(feedDir + file)
    plt.imshow(image)
    plt.pause(0.05)

    # read in input to move file into correct folder.
    while True:
        x = getch.getch()
        if x == 'f':
            os.rename(feedDir + file, peopleDir + file)
            lastAnswer = peopleDir
            print('moved to people')
            break
        elif x == 'j':
            os.rename(feedDir + file, noPeopleDir + file)
            lastAnswer = noPeopleDir
            print('moved to noPeople')
            break
        elif x == 'b':
            if lastFile == None:
                print('Can not go back more than 1')
            else:
                os.rename(lastAnswer + lastFile, feedDir + lastFile)
                lastAnswer == None
                tempLastFile = lastFile
                lastFile == None
                labelSingleImage(tempLastFile)
    lastFile = file




if (len(sys.argv) != 4):
    print("USAGE: python labelToolTest [feedDir] [peopleDir] [noPeopleDir]")
else:
    feedDir = sys.argv[1]
    peopleDir = sys.argv[2]
    noPeopleDir = sys.argv[3]

    # read in all png files
    imageFiles = glob.glob(feedDir + '*.png')



    # go through every file and classify image
    for file in imageFiles:
        labelSingleImage(file)

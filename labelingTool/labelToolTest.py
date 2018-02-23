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


# read in all png files
imageFiles = glob.glob('*.png')

lastFile = None
lastAnswer = None

def labelSingleImage(file):
    global lastFile
    global lastAnswer
    print('f - people, j - no people, b - back')

    image = im.imread(file)
    plt.imshow(image)
    plt.pause(0.05)

    # read in input to move file into correct folder.
    while True:
        x = getch.getch()
        if x == 'f':
            os.rename(file, 'people/' + file)
            lastAnswer = 'people/'
            print('moved to people')
            break
        elif x == 'j':
            os.rename(file, 'noPeople/' + file)
            lastAnswer = 'noPeople/'
            print('moved to noPeople')
            break
        elif x == 'b':
            if lastFile == None:
                print('Can not go back more than 1')
            else:
                os.rename(lastAnswer + lastFile, lastFile)
                lastAnswer == None
                tempLastFile = lastFile
                lastFile == None
                labelSingleImage(tempLastFile)
    lastFile = file

# go through every file and classify image
for file in imageFiles:
    labelSingleImage(file)

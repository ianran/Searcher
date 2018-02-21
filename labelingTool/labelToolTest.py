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

print('f - people, j - no people')

# go through every file and classify image
for file in imageFiles:
    image = im.imread(file)
    plt.imshow(image)
    plt.pause(0.05)

    # read in input to move file into correct folder.
    while True:
        x = getch.getch()
        if x == 'f':
            os.rename(file, 'people/' + file)
            break
        elif x == 'j':
            os.rename(file, 'noPeople/' + file)
            break

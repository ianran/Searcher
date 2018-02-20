# Note needs getch python package

import numpy as np
import scipy.ndimage as im
import matplotlib.pyplot as plt
import getch
import glob
import os

#x = im.imread('DesertRocks.png')
#plt.imshow(x)
#plt.show()

imageFiles = glob.glob('*.png')

print('f - people, j - no people')

#plt.axis()
plt.ion()


for file in imageFiles:
    image = im.imread(file)
    plt.imshow(image)
    #plt.draw()
    plt.pause(0.05)

    while True:
        x = getch.getch()
        if x == 'f':
            os.rename(file, 'people/' + file)
            break
        elif x == 'j':
            os.rename(file, 'noPeople/' + file)
            break

# Note needs getch python package
# https://pypi.python.org/pypi/getch#downloads
#
# USAGE
# place image files in same folder, must also have folders named
# people/ and noPeople/

import numpy as np
import scipy.ndimage as im
import matplotlib.pyplot as plt
import glob
import os
import sys
import platform

if platform.system() == 'Windows':
    from msvcrt import getch
else:
    import getch

# The directory to read images from
feedDir = None
# no people, and people directories
noPeopleDir = None
peopleDir = None

# variables to store what the last file and answer given were
# used to allow you to go back and re label the last image if you pressed the wrong button.
lastFile = None
lastAnswer = None


# labelSingleImage
# this function
def labelSingleImage(file):
    global lastFile
    global lastAnswer
    print('p - people, n - no people, f - back (I Fucked up button)')

    # read in just the base file name, removes path data
    file = os.path.basename(file)

    # reads in image file and displays image
    image = im.imread(feedDir + file)
    plt.imshow(image)
    plt.pause(0.05)

    # Read in input to move file into correct folder.
    while True:
        # read in character for terminal without return statement
        x = getch.getch()

        if x == 'p':
            # there are people, move to the people directory
            os.rename(feedDir + file, peopleDir + file)
            lastAnswer = peopleDir
            print('moved to people')
            break

        elif x == 'n':
            # there are no people,  move to the no people directory
            os.rename(feedDir + file, noPeopleDir + file)
            lastAnswer = noPeopleDir
            print('moved to noPeople')
            break

        elif x == 'f':
            # the go back button, if there is not a lastFile set to go back, tell
            # the user they can
            if lastFile == None:
                print('Can not go back more than 1')
            else:
                # if there is a last file to go back to
                # move that file back to the feed directory, set last file to be none,
                # and recursively call labelSingleImage again on the last file.
                os.rename(lastAnswer + lastFile, feedDir + lastFile)
                lastAnswer = None
                tempLastFile = lastFile
                lastFile = None
                labelSingleImage(tempLastFile)
            # End if
        # End if
    # End while

    lastFile = file

# End labelSingleImage(file)


# check if all arguments are given, and output usage if not
if (len(sys.argv) != 4):
    print("USAGE: python labelToolTest [feedDir] [peopleDir] [noPeopleDir]")
else:
    # define input arguments to variables

    feedDir = sys.argv[1]
    peopleDir = sys.argv[2]
    noPeopleDir = sys.argv[3]
# End if

# Read in all jpg file names
imageFiles = glob.glob(feedDir + '*.jpg')



# Go through every file and classify image
for file in imageFiles:
    labelSingleImage(file)

# End for

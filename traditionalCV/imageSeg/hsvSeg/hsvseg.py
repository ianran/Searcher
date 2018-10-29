import cv2
import numpy as np
import math
from enum import Enum
from matplotlib import image as mpimage
from matplotlib import pyplot as plt
import glob
import csv
from progressbar import ProgressBar
import sys
import shutil

class OrangeHSVSeg:
    """
    An OpenCV pipeline generated by GRIP.
    """
    
    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__hsv_threshold_hue = [0, 30]
        self.__hsv_threshold_saturation = [60, 255]
        self.__hsv_threshold_value = [220, 255]

        self.hsv_threshold_output = None


    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step HSV_Threshold0:
        self.__hsv_threshold_input = source0
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)


    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_RGB2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))


def ROC(lab, genLab, thresh):

    TPR = 0
    TNR = 0
    FPR = 0
    FNR = 0

    TPC = 0.0
    FPC = 0.0
    TNC = 0.0
    FNC = 0.0
 
    for i, val in genLab.items():

        if val == True and lab[i] == True:
            TPC += 1
        elif val == True and lab[i] == False:
            FPC += 1
        elif val == False and lab[i] == False:
            TNC += 1
        elif val == False and lab[i] == True:
            FNC += 1
        else:
            print('problem')
        # End if

    # End for

    # calculate rates
    TPR = TPC/(TPC + FNC)
    FPR = FPC/(FPC + TNC)
    TNR = TNC/(FPC + TNC)
    FNR = FNC/(TPC + FNC)

    return TPR, FPR, TNR, FNR

# End ROC

def isOrange(segIm, thresh):

    count = np.count_nonzero(segIm)

    if count >= thresh:
        return(True)
    else:
        return(False)
    # End if

# End isOrange

if len(sys.argv) != 4:

    print("Usage: python hsvSeg [imgDir] [csvFile] [pedictedOrangeDirectory]")
    sys.exit()

# End if

imgDir = sys.argv[1]
csvFile = sys.argv[2]
predDir = sys.argv[3]

threshold = 15

# Read in the labels for images
reader = csv.reader(open(csvFile))

labels = {}
genLabels = {}

for row in reader:

    key = row[0]

    if row[1] == 'people':
        labels[key] = True
    else:
        labels[key] = False
    # End if

# End for

# Create system pipeline.
segPipe = OrangeHSVSeg()

pBar = ProgressBar()

# Read images one at a time and put them through color segmentation.
for fName in pBar(glob.glob(imgDir + "**/*.jpg")):

    splitDir = fName.split('/')
    key = splitDir[len(splitDir) - 1]
    
    genLabels[key] = []

    image = mpimage.imread(fName)

    # Segment image
    segPipe.process(image)
    segImage = segPipe.hsv_threshold_output
    
    # Determine label based on threshold
    genLabels[key] = isOrange(segImage, threshold)

    if genLabels[key] == True:

        shutil.copyfile(fName, predDir + key)

    # End if

# End for

TPR, FPR, TNR, FNR = ROC(labels, genLabels, threshold)

print("TPR: ", TPR)
print("FPR: ", FPR)
print("TNR: ", TNR)
print("FNR: ", FNR)

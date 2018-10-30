import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import shutil
import sys
sys.path.append('../../../labelingTool/')
sys.path.append('../../../UI/')
import labelReader
import ImageDisplayer
from enum import Enum
from mpl_toolkits import mplot3d
from progressbar import ProgressBar

# Class for RGB segmentation for orange
class OrangeSegRGB:
	"""
	An OpenCV pipeline generated by GRIP.
	"""

	def __init__(self):
		"""initializes all values to presets or None if need to be set
		"""

		self.__rgb_threshold_red = [178.86690647482015, 255.0]
		self.__rgb_threshold_green = [0.0, 160.55555555555554]
		self.__rgb_threshold_blue = [0.0, 212.07070707070707]

		self.rgb_threshold_output = None


	def process(self, source0):
		"""
		Runs the pipeline and sets all outputs to new values.
		"""
		# Step RGB_Threshold0:
		self.__rgb_threshold_input = source0
		(self.rgb_threshold_output) = self.__rgb_threshold(self.__rgb_threshold_input, self.__rgb_threshold_red, self.__rgb_threshold_green, self.__rgb_threshold_blue)


	@staticmethod
	def __rgb_threshold(input, red, green, blue):
		"""Segment an image based on color ranges.
		Args:
			input: A BGR numpy.ndarray.
			red: A list of two numbers the are the min and max red.
			green: A list of two numbers the are the min and max green.
			blue: A list of two numbers the are the min and max blue.
		Returns:
			A black and white numpy.ndarray.
		"""
		# Replace below if images are read in thorugh opencv
		# out = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
		# return cv2.inrange(out, (red[0], green[0], blue[0]), (red[1], green[1], blue[1]))

		return cv2.inRange(input, (red[0], green[0], blue[0]),  (red[1], green[1], blue[1]))


# Function to calculate True Positive and False Positive Rate by counting number of true positives,
# false positives, true negatives, and false negatives
# Input:
#	true -- Dictionary with true labels
#	pred -- Dictionary with predicted labels
# Output:
#	tpr -- Array with true positive rates for thresholds
# 	fpr -- Array with false positive rates for thresholds
def findTPRandFPR(true, pred):
	# Create progress bar for finding rates
	print('TPR and FPR progress:')
	prog2 = ProgressBar()
	
	# Initialize rate arrays
	tpr = 0.
	fpr = 0.
	
	tp = 0.
	fp = 0.
	tn = 0.
	fn = 0.
	
	# Go through each image's labels
	for fName in prog2(pred.keys()):

		# True positive
		if true.get(fName) == 'people' and pred.get(fName) == True:
			tp += 1
		# False negative
		elif true.get(fName) == 'people' and pred.get(fName) == False:
			fn += 1
		# False positive
		elif true.get(fName) == 'noPeople' and pred.get(fName) == True:
			fp += 1
		# True negative
		elif true.get(fName) == 'noPeople' and pred.get(fName) == False:
			tn += 1

	# Calculate true positive rate
	tpr = tp / (tp + fn)

	# Calculate false positive rate
	fpr = fp / (fp + tn)

	return tpr, fpr


# check if all arguments are given, and output usage if not
if (len(sys.argv) != 4):
	# print usage, and exit program
	print("USAGE: python orangeRGBseg [imgDir] [csvFile] [predOrange]")
	sys.exit()
#end if


# define input arguments to variables
imgDir = sys.argv[1]
csvFile = sys.argv[2]
predDir = sys.argv[3]


# Dispalyer object initialization
display = ImageDisplayer.ImageDisplayer()

# Read in image labels
labels = labelReader.readLabelsDict(csvFile)


# Initialize dict for predicted labels
orangeDict = {}


# Image characteristics
imgSize = (1530, 2720, 3)
img = np.zeros(imgSize)
segImg = np.zeros((imgSize[0], imgSize[1]))


# Create RGB Segmenter object
seg = OrangeSegRGB()


# Threshold
thres = 20


# Create progress bar for image segmentation
prog1  = ProgressBar()
print('Segmentation progress:')


# Read images one at a time and put them through color segmentation.
for fName in prog1(glob.glob(imgDir + '**/*.jpg')):
	# Key for label dict is image name
	fNameSplit = fName.split('/')
	key = fNameSplit[len(fNameSplit)-1]

	orangeDict[key] = []

	# Read in image
	img = mpimg.imread(fName)

    	# Segment image
	seg.process(img)
	segImg = seg.rgb_threshold_output

   	# Determine number of orange pixels segmented
	orangePixNum = np.count_nonzero(segImg)

	# Label as True and copy image to directory containing predicted orange pictures
	if orangePixNum >= thres:
		orangeDict[key] = True
		shutil.copyfile(fName, predDir + key)
	# Label as False
	else:
		orangeDict[key] = False


# Calculate True Positive Rate and False Positive Rate
tpr,fpr = findTPRandFPR(labels, orangeDict)

print('')
print('TPR = ', tpr)
print('FPR = ', fpr)
print('')


# Add path for predicted orange images
display.addImgFiles(glob.glob(predDir + '*.jpg'))

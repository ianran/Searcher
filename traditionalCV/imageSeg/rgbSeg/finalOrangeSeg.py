#########################################################################################
# orangeRGBseg.py									#
#											#
# Joellen Lansford									#
#											#
# This program reads in a directory of images and a csv file of image labels, performs 	#
# RGB segmentation on them, moves images with orange pixels to a new directory, 	#
# displays the images with orange, and calculates TPR, FPR, TNR, and FNR.		#
#########################################################################################
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import shutil
import scipy.misc
import PIL.Image
from PIL.ExifTags import TAGS
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

		return cv2.inRange(input, (red[0], green[0], blue[0]),  (red[1], green[1], blue[1]))


# Function to calculate True Positive, False Positive, True Negative, and False Negative Rate by counting number of true positives,
# false positives, true negatives, and false negatives
# Input:
#	true -- Dictionary with true labels
#	pred -- Dictionary with predicted labels
# Output:
#	tpr -- True positive rate
# 	fpr -- False positive rate
# 	tnr -- True negative rate
# 	fnr -- False negative rate
def findRates(true, pred):
	# Create progress bar for finding rates
	print('TPR and FPR progress:')
	prog2 = ProgressBar()
	
	# Initialize rate arrays
	tpr = 0.
	fpr = 0.
	tnr = 0.
	fnr = 0.
	
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

	# Calculate true negative rate
	tnr = tn / (fp + tn)

	# Calculate false negative rate
	fnr = fn / (fn + tp)

	print('TP = ', tp)
	print('FP = ', fp)
	print('TN = ', tn)
	print('FN = ', fn)

	return tpr, fpr, tnr, fnr


# Function to find and draw contours of orange pixels found in images
# Input:
#	img -- RGB image matrix
#	segImg -- Binary segmented image matrix
# Output:
#	highImg -- RGB highlighted iamge matrix
def POI(img, segImg):
	im2, contours, hierarchy = cv2.findContours(segImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	highImg = cv2.drawContours(img, contours, -1, (0,0,255), 2)

	return highImg


# check if all arguments are given, and output usage if not
if (len(sys.argv) != 4):
	# print usage, and exit program
	print("USAGE: python orangeRGBseg.py [imgDir] [csvFile] [predOrange]")
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
i = 0
for fName in prog1(glob.glob(imgDir + '*.jpg')):
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
		metadata = (PIL.Image.open(fName))._getexif()

		metaTags = {}

		for tag, val in metadata.items():

			metaTags[TAGS.get(tag, tag)] = val

		latMeta = metaTags['GPSInfo'][2]
		longMeta = metaTags['GPSInfo'][4]
		latDeg = latMeta[0][0] / latMeta[0][1]
		latMin = latMeta[1][0] / latMeta[1][1]
		latSec = latMeta[2][0] / latMeta[2][1]
		longDeg = longMeta[0][0] / longMeta[0][1]
		longMin = longMeta[1][0] / longMeta[1][1]
		longSec = longMeta[2][0] / longMeta[2][1]
		print(key + '- ', 'Lat: ', latDeg, '\N{DEGREE SIGN}, ', latMin, '\', ', latSec, '\" ', 'Long: ', longDeg, '\N{DEGREE SIGN}, ', longMin, '\', ', longSec, '\"')
		highImg = POI(img, segImg)
		saveImg = cv2.cvtColor(highImg, cv2.COLOR_RGB2BGR)
		cv2.imwrite(predDir + key, saveImg)
		#shutil.copyfile(fName, predDir + key)
	# Label as False
	else:
		orangeDict[key] = False

	i += 1

print('')
print('Num of images: ', i)

# Calculate True Positive Rate and False Positive Rate
#tpr,fpr,tnr,fnr = findRates(labels, orangeDict)

#print('')
#print('TPR = ', tpr)
#print('FPR = ', fpr)
#print('TNR = ', tnr)
#print('FNR = ', fnr)
#print('')


# Add path for predicted orange images
display.addImgFiles(glob.glob(predDir + '*.jpg'))

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import sys
sys.path.append('../../../labelingTool/')
import labelReader
from sklearn import metrics
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


# Function calculate True Positive and False Positive Rate by counting number of true positives,
# false positives, true negatives, and false negatives
# Input:
#	true -- Dictionary with true labels
#	pred -- Dictionary with predicted labels
#	numThres -- Number of threshold values sweeped over for predicted labels
# Output:
#	tpr -- Array with true positive rates for thresholds
# 	fpr -- Array with false positive rates for thresholds
def findTPRandFPR(true, pred, numThres):
	# Create progress bar for finding rates
	print('TPR and FPR progress:')
	prog2 = ProgressBar()
	
	# Initialize rate arrays
	tpr = np.zeros(numThres)
	fpr = np.zeros(numThres)
	
	# Sweep through thresholds
	for i in prog2(range(numThres)):
		# Initialize outcome counts
		tp = 0.
		fp = 0.
		tn = 0.
		fn = 0.

		# Go through each image's labels
		for fName in pred.keys():

			# True positive
			if true.get(fName) == 'people' and pred.get(fName)[i] == True:
				tp += 1
			# False negative
			elif true.get(fName) == 'people' and pred.get(fName)[i] == False:
				fn += 1
			# False positive
			elif true.get(fName) == 'noPeople' and pred.get(fName)[i] == True:
				fp += 1
			# True negative
			elif true.get(fName) == 'noPeople' and pred.get(fName)[i] == False:
				tn += 1

		# Calculate true positive rate
		tpr[i] = tp / (tp + fn)

		# Calculate false positive rate
		fpr[i] = fp / (fp + tn)

	return tpr, fpr


# check if all arguments are given, and output usage if not
if (len(sys.argv) != 3):
	# print usage, and exit program
	print("USAGE: python orangeRGBseg [imgDir] [csvFile]")
	sys.exit()
#end if

# define input arguments to variables
imgDir = sys.argv[1]
csvFile = sys.argv[2]


# Read in image labels
labels = labelReader.readLabelsDict(csvFile)


# Initialize dict for predicted labels
orangeDect = {}


# Image characteristics
imgSize = (1530, 2720, 3)
img = np.zeros(imgSize)
segImg = np.zeros((imgSize[0], imgSize[1]))


# Create RGB Segmenter object
seg = OrangeSegRGB()
print('Segmentation progress:')

# Thresholds
thres = np.arange(0,2001,10)


# Create progress bar for image segmentation
prog1  = ProgressBar()


# Read images one at a time and put them through color segmentation.
for fName in prog1(glob.glob(imgDir + '*/*.jpg')):
	# Key for label dict is image name
	fNameSplit = fName.split('/')
	key = fNameSplit[len(fNameSplit)-1]

	orangeDect[key] = []

	# Read in image
	img = mpimg.imread(fName)

    	# Segment image
	seg.process(img)
	segImg = seg.rgb_threshold_output

   	# Determine number of orange pixels segmented
	orangePixNum = np.count_nonzero(segImg)

	# Run image through all thresholds
	t = 0
	for th in thres:
		# Label as True
		if orangePixNum >= th:
			orangeDect[key].append(True)
		# Label as False
		else:
			orangeDect[key].append(False)


# Calculate ROC
tpr,fpr = findTPRandFPR(labels, orangeDect, np.size(thres))

print(tpr[1], fpr[1], thres[1])

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve', marker='o')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.xticks(np.arange(0,1.1,0.05))
plt.ylim([0.0, 1.05])
plt.yticks(np.arange(0,1.1,0.05))
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")



# Plot tpr vs fpr vs threshold
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(tpr[1:21], fpr[1:21], thres[1:21], c='r', marker='o')
ax.set_xlabel('True Positive Rate')
ax.set_zlabel('False Positive Rate')
ax.set_zlabel('Threshold')
plt.show()
# csvToFolders
# Written Ian Rankin March 2018
# This script will read in the csv file of all of the labels, and
# move files from a feed directory into people, and noPeople directories.
#
#
# USAGE:
# python csvToFolders.py
#

import sys
import os
import labelReader as reader
import glob

# check all arguments have been passed
if (len(sys.argv) != 3):
    # print usage, and exit program
    print('USAGE python fileFolderToCSV [feedDir] [csvFile]')
    sys.exit()

# create directories if they don't already exist.
os.system('mkdir people')
os.system('mkdir noPeople')

feedDir = sys.argv[1]
csvFile = sys.argv[2]

imageFiles = glob.glob(feedDir + '*.jpg')

labels = reader.readLabelsDict(csvFile)

# go through every file and move it based off what was in the
# csv file.
for file in imageFiles:
    if labels[os.path.basename(file)] == 'people':
        os.rename(file, 'people/' + os.path.basename(file))
    elif labels[os.path.basename(file)] == 'noPeople':
        os.rename(file, 'noPeople/' + os.path.basename(file))
    else:
        print('BAD LABEL Can not place!')
print('Completed successfully')

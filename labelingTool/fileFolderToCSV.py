# fileFolderToCSV
# Written Ian Rankin March 2018
# This tool is designed when given a list of jpg's in either
# the people or no people folder into a CSV file formated as
# [filename], [people | noPeople]
# ex:
#
# file1.jpg, people
# file2.jpg, noPeople
#
# This will allow us to combine all of the data together
# into a single csv that can be stored on the git repo
#
# USAGE
# python fileFolderToCSV [peopleDir] [noPeopleDir]

import glob
import os
import sys
import labelReader as reader

# check all arguments have been passed
if (len(sys.argv) != 4):
    # print usage, and exit program
    print('USAGE python fileFolderToCSV [peopleDir] [noPeopleDir] [csvFile]')
    sys.exit()

# define input arguments
peopleDir = sys.argv[1]
noPeopleDir = sys.argv[2]

# filename reading
peopleImageNames = glob.glob(peopleDir + '*.jpg')
noPeopleImageNames = glob.glob(noPeopleDir + '*.jpg')

# read in current csv, and make sure not overiding files
# already in csv file.
curLabels = reader.readLabelsDict(sys.argv[3])

# open csv file for appending data
with open(sys.argv[3], 'a') as f:
    for p in peopleImageNames:
        if not (os.path.basename(p) in curLabels):
            f.write(os.path.basename(p) + ',people\n')
    for p in noPeopleImageNames:
        if not (os.path.basename(p) in curLabels):
            f.write(os.path.basename(p) + ',noPeople\n')

# labelReader.py
# Written Ian Rankin March 2018
# This is a file containing the function(s) to access labeled data
# easier
# These are various easy to use functions for reading in the data
#
#
# USAGE (example):
# import labelReader
#
# labelReader.readLabels('labels.csv')

import csv

# readLabelsDict
# This fucntion reads the csv file given, and returns a dict
# of the labels with filename (with .jpg)
# The format of the file must be in:
# filename, label
# filename, label
#
# @param labelFilename - the name of the file with labels you are reading in.
# @param returns - directory of labels, empty if file doesn't exist.
def readLabelsDict(labelFilename):
    labels = {}

    try:
        with open(labelFilename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                labels[row[0]] = row[1]
    except FileNotFoundError:
        print('readLabelsDict can not read ' + labelFilename + ' returning empty dict')

    return labels

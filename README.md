![Searcher Logo](https://github.com/ianran/Searcher/blob/master/SearcherLogo.png)

Searcher is a project to aid search and rescue operations by autonomously flying an unmanned vehicle and detect where a person is. Unmanned Aerial Vehicles (UAVs) are already in use by search and rescue teams to search for people in hard to reach locations, but all of their drones require somebody to actively fly and watch video feed. We propose a system that autonomously flies and determines if a person is in the search area using machine vision and machine learning. We plan on using a drone for this project with a camera, and possibly an infrared camera to pick up heat signatures. Basic success would require developing an algorithm which would automatically detect if there is a person in a aerial image with 80% TPR and 10% FPR, and integrate the algorithm into a drone in near real-time. Near real-time is defined as the processing of frames coming once every few seconds, which would still be possible and useful if the UAV is flying fairly slowly. Stretch goals would be to have multiple autonomous UAVs working together to search for a person with very high accuracy, ~95% with few false positives. Also, adding a feature that would allow the human operators to verify if a person identified by the system is the person they are looking for.

Folders:

### CNN - Has code for CNN networks and training.
Includes both an older CNN and the latest traditional CNN and CGAN code in folder called CGAN.



### labelingTool - contains code for labeling data and pre processing

csvToFolder.py - takes images in a folder, and moved them to other folders based on the labels in the labeled csv folder.

FeedImagesToNumpy.py - takes images in a folder and compresses them for reading in easily in python. It also reads in labels and creates a one-hot encoded vector for the images as well.

fileFolderToCSV.py - Takes images in different folders as returned by the labeling tool and places the labels into a csv file.

labelReader.py - basic python code to read csv file and use the given labels as a python dictionary.

labelToolTest.py - shows images from a folder and allows you to label the image which moves them into directories based on the label.

labels.csv - Giant csv of all labels for our dataset.

videoExtractor.sh - a bash script file to allow easy extraction of images from a video file using ffmpeg.

### retraining - example code from tensorflow used to retrain last layer of inception, did not work.

### traditionalCV - code for segmentation and older optical flow effort.

#### imageSeg - image segmentation code used for final output.

#### opticalFlow - old code for doing optical flow.

### UI - user interface code for displaying images after processing


List of video files:
## AMountain videos

AMountain1.MOV - labeled

AMountain2.MOV - labeled

## Animas1 video

Animas1_0004.MOV - labeled

Animas1_0008.MOV - Labeled

Animas1_0009.MOV - labeled

## AnimasV2 video

AnimasV2???? (Must be on my desktop only???)

## Cascade Creek Video

CascadeCreek_0001.MOV - labeled

CascadeCreek_0002.MOV - labeled

CascadeCreek_0003.MOV - labeled

## FallsCreek Video (lots of data)

FallsCreek_0012.MOV - labeled

FallsCreek_0014.MOV - labeled


## GrassyField Videos (Should be renamed before being labeled to avoid naming conflicts)

xxxxxx_0003.MOV - labeled

xxxxxx_0009.MOV - labeled


## HorseShoeData

DJI_0001.MOV - Don't Bother (or label as all not people)

DJI_0002.MOV - Don't bother (bad data)

DJI_0004.MOV - Don't bother (bad data)

DJI_0005.MOV - labeled

DJI_0006.MOV - labeled

DJI_0007.MOV - Don't bother

DJI_0008.MOV - labeled

DJI_0011.MOV - not labeled (really short)

DJI_0012.MOV - Don't bother

DJI_0013.MOV - Don't bother

## Narnia Data

DJI_0014.MOV - labeled

DJI_0015.MOV - labeled

DJI_0016.MOV - labled

DJI_0017.MOV - don't bother

DJI_0018.MOV - labeled

## StochajData

DJI_0040.MOV - labeled

DJI_0043.MOV - labeled

DJI_0044.MOV - labeled








Note:  
Let's try to keep master as completely working code.  
For development branch off your own branch and get it working  
before merging the master with that branch.  


# Git links

To install git:
https://git-scm.com

Intro tutorial for git:
https://www.git-scm.com/docs/gittutorial




Git cheat sheet:  
[] = arguments  
// = comments on commands  

# List of useful commands

git status                          // Tells you changed files and staged files.  
                                    // Staged files in green, unstaged files in red.  
git add [files to add to staging]   // moves files to add to staging to staged  
git add --all                       // moves all changed files to staged  
git commit -m "[commit message]"    // Commits all staged files with commit message.  
git commit                          // Commits, and starts default text editor for commit message  
                                        // Note: Default editor is often vi which can be closed with esc :x  
git push                            // Pushes local commits to server  
git pull                            // Pulls data from server, also merges at same time.  

git fetch                           // Generic pulls data from server, does not merge.  
git log                             // Shows list of recent commits.  
git checkout [branch to checkout]   // Changes your repo to a different branch, note: can go to any arbitrary commit  
git branch -v                       // Shows list of branches  
git branch [branch name]            // Creates a new branch at the current branch.  
git merge [branch name]             // Merges your current branch with [branch name]  
git stash                           // Doesn't commit, but cleans working branch. (ONLY use if you want to get rid of current  
                                        unstaged data, after checking out another commit) (I don't know exactly what it does to be honest)  


# Generic commit:

git add [files to commit]  
git commit -m "[commit message]"  
git pull  
git push  


# Starting new development branch:

git checkout master  
git branch [dev branch name]  
git checkout [dev branch name]  
...  // various commits until code is working, may need to merge with somebody else's branch as well.  
git merge master  
... // make sure code is still working with latest master branch  
git status // MAKE SURE that working branch is clean (don't need to commit)  
git checkout master  
git merge [dev branch name]  
    // may need to commit again.  
git branch -d [dev branch name]  

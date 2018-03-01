# The video extractor is a simple script to extract video data
# For those who are using bash and have ffmpeg installed, should
# be very easy to use.
#
# USAGE: 'sh videoExtractor [videoFileToExtract]
#
# you can also just copy the last command and replace $1 with the video
# filename.

# performs check to make sure correct number of arguements passed
if [ $# -ne 1 ]
    then
        echo "USAGE: sh videoExtractor [videoFileToExtract]"
        exit -1
fi

# places into variable name just the single filename without extension
filename=$1
name=${filename%.*}
echo $name

# run ffmpeg command
ffmpeg -i $1 -r 2 $name-%05d.jpg

# ImageDisplayer.py
# Written Ian Rankin September 2018
#
# This code takes as an input images that have been processed to think
# There might be a person in the image, and shows them to the user in
# another thread to allow new images to still be processed in the main
# thread of execution
#
# TODO: add method to show GPS coordinate along with file.

import threading
import scipy.ndimage as im
import matplotlib.pyplot as plt
import platform
import time
if platform.system() == 'Windows':
    from msvcrt import getch
else:
    import getch



class ImageDisplayer:
    # run
    # threaded run function for showing images.
    def run(self):
        while True:
            image = im.imread(self.files[self.idx])
            plt.clf()
            plt.axis('off')
            plt.title(self.files[self.idx])
            print('trying to show image')
            plt.imshow(image)
            print('Should have shown image')
            plt.pause(0.05)

            char = 'p'
            while not(char == 's' or char == 'w' or char == 'q'):
                char = getch.getch()
                print(char)
            print('Yo')
            if char == 's':
                if self.idx > 0:
                    self.idx -= 1
            elif char =='w':
                if self.idx >= len(self.files):
                    self.idx += 1
            else:
                return

    def __init__(self):
        self.idx = 0
        self.files = []
        self.running = False

        self.thread = threading.Thread(target=self.run)#, args=[self])

    # addImgFiles
    # adds image files, and starts the thread if it is the first image file added image.
    # @param newFiles
    def addImgFiles(self, newFiles):
        if len(newFiles) > 0:
            self.files.extend(newFiles)
            if not self.running:
                self.thread.start()



display = ImageDisplayer()
time.sleep(1)
display.addImgFiles(['../data/grace_hopper.jpg'])
time.sleep(40)

# ImageDisplayer.py
# Written Ian Rankin September 2018
#
# This code takes as an input images that have been processed to think
# There might be a person in the image, and shows them to the user in
# another thread to allow new images to still be processed in the main
# thread of execution
#
# # example usage
# display = ImageDisplayer()
# files = glob.glob('../labelingTool/feed/*.jpg')
# display.addImgFiles(files)
#
# TODO: add method to show GPS coordinate along with file.

import threading
import scipy.ndimage as im
import glob
#import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageTk, Image
import platform
import time
if platform.system() == 'Windows':
    from msvcrt import getch
else:
    import getch




class ImageDisplayer:
    # run
    # threaded run function for showing images.
    # Should not be called directly.
    def run(self):
        # create tkinter root and open an image.
        root = tk.Tk()
        image = Image.open(self.files[self.idx])
        image.thumbnail(self.displaySize)
        photo = ImageTk.PhotoImage(image)

        # Create label with image.
        label = tk.Label(root, image=photo, text="Place GPS Coordinates here.", compound=tk.BOTTOM)
        label.image = photo
        label.pack()

        # update the display
        root.update_idletasks()
        root.update()

        while True:
            # Open new image
            im = Image.open(self.files[self.idx])
            im.thumbnail(self.displaySize)
            photo = ImageTk.PhotoImage(im)

            # configure the label for updating graphics
            label.configure(image=photo, text="Place GPS coordinates here i = " + str(self.idx))
            label.image=im

            # update graphics
            root.update_idletasks()
            root.update()

            ########### Handle reading keyboard input.
            char = 'p'
            # if a bad character, wait until a good character is given before doing anything.
            while not(char == 's' or char == 'w' or char == 'q'):
                char = getch.getch()

            if char == 's':
                if self.idx > 0:
                    self.idx -= 1
                else:
                    print('Out of images going backwards.')
            elif char =='w':
                if self.idx < len(self.files)-1:
                    self.idx += 1
                else:
                    print('No more images processed so far.')
            else:
                return

    # inits the object.
    def __init__(self):
        self.idx = 0
        self.files = []
        self.running = False
        # Change display size by modifying these settings.
        self.displaySize = (544,306)

        self.thread = threading.Thread(target=self.run)#, args=[self])
        print('Press w to feed forward through images.')
        print('Press s to go back through images.')
        print('Press q to quit thread')


    # addImgFiles
    # adds image files, and starts the thread if it is the first image file added image.
    # @param newFiles
    def addImgFiles(self, newFiles):
        if len(newFiles) > 0:
            self.files.extend(newFiles)
            if not self.running:
                self.thread.start()

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
    def run(self):
        print('Hello')

    def __init__(self):
        self.files = []
        self.running = False

        self.thread = threading.Thread(target=self.run, args=(self))

    def addImgFiles(self, newFiles):
        self.files.append(newFiles)
        if not self.running:
            self.thread.start()



display = ImageDisplayer()
time.sleep(1)
display.addImgFiles([])

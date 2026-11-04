#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_Core04b_ThresholdLoop.py   (na pasta do PL05.zip)
# O mesmo limiar com dois ciclos, com NumPy e com o OpenCV, e o tempo de cada um.

import cv2, time
import numpy as np
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent
################# Loop ####################################
def thresholdUsingLoop(src, thresh, maxValue):
    # Create a output image
    dst = src.copy()
    height, width = src.shape[:2]

    # Loop over rows
    for i in range(height):
        # Loop over columns
        for j in range(width):
            if src[i, j] > thresh:
                dst[i, j] = maxValue
            else:
                dst[i, j] = 0

    return dst

################# Vectorized ###########################
def thresholdUsingVectors(src, thresh, maxValue):
    # Create a black output image ( all zeros )
    dst = np.zeros_like(src)

    # Find pixels which have values>threshold value
    thresholdedPixels = src > thresh

    # Assign those pixels maxValue
    dst[thresholdedPixels] = maxValue

    return dst
#######################################################
imagePath = str(PASTA / "imgs" / "astronaut.png")
# Read image in Grayscale format
testImage = cv2.imread(imagePath,0)
# Set threshold and maximum value
thresh = 100
maxValue = 255
#--------------------
t = time.time()
dst = thresholdUsingLoop(testImage, thresh, maxValue)
print("Time loop = {} seconds".format(time.time() - t))
#-------------------
t = time.time()
dst = thresholdUsingVectors(testImage, thresh, maxValue)
print("Time Vectorized = {} seconds".format(time.time() - t))
#---------OpenCV
t = time.time()
th, dst = cv2.threshold(testImage, thresh, maxValue, cv2.THRESH_BINARY)
print("Time OpenCV = {} seconds".format(time.time() - t))

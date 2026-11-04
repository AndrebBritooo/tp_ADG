#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6", "matplotlib"]
# ///
# Correr com: uv run exemplos/IP_Core11_Histogram.py   (na pasta do PL05.zip)
# Histograma com OpenCV, NumPy e Matplotlib.

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Load the image
img_bgr = cv.imread(str(PASTA / "imgs" / "astronaut.png"))
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# Calculate the histogram using OpenCV and NumPy
hist_ocv = cv.calcHist([img_gray], [0], None, [256], [0, 256])
hist_np, _ = np.histogram(img_gray.ravel(), 256, [0, 256])

# ... do something with the histogram (e.g. print the values)
print(hist_ocv)
print(hist_np)

# Plot the histogram using Matplotlib
plt.hist(img_gray.ravel(), 256, range=[0, 256])
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Pixel Count")
plt.show()

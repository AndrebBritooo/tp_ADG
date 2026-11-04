#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6", "matplotlib"]
# ///
# Correr com: uv run exemplos/IP_Core12_HistEqualiz.py   (na pasta do PL05.zip)
# Equalização do histograma.

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Load the image and convert it to grayscale
img_bgr = cv.imread(str(PASTA / "imgs" / "astronaut.png"))
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# Calculate the histogram of the original image
hist_orig = cv.calcHist([img_gray], [0], None, [256], [0, 256])
print("Original histogram:", hist_orig.ravel())

# Apply histogram equalization to the grayscale image
img_eq = cv.equalizeHist(img_gray)

# Calculate the histogram of the equalized image
hist_eq = cv.calcHist([img_eq], [0], None, [256], [0, 256])
print("Equalized histogram:", hist_eq.ravel())

# Display the original and equalized images side-by-side
res = np.hstack((img_gray, img_eq))
cv.imshow("Original vs. Equalized", res)

# Plot the original and equalized histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.hist(img_gray.ravel(), 256, range=[0, 256], color="blue")
ax1.set_title("Original Histogram")

ax2.hist(img_eq.ravel(), 256, range=[0, 256], color="red")
ax2.set_title("Equalized Histogram")

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()

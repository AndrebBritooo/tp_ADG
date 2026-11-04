#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core01c_ImgPerspective.py   (na pasta do PL05.zip)
# Perspetiva.

import cv2 as cv
import numpy as np
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Load the image
img = cv.imread(str(PASTA / "imgs" / "astronaut.png"))
assert img is not None, "Image not found"

rows, cols = img.shape[:2]

# Define the four corners of the region of interest
# The region of interest is the part of the image that we want to transform
# In this case, we define the four corners of the original image
src = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

# Define the four corners of the desired output
dst = np.float32([[100, 150], [300, 180], [150, 250], [350, 350]])

# The perspective transform matrix is computed using the source and destination points
transform_matrix = cv.getPerspectiveTransform(src, dst)
result = cv.warpPerspective(img, transform_matrix, (cols, rows))

# Display the result
cv.imshow("Original", img)
cv.imshow("Result", result)

cv.waitKey(0)
cv.destroyAllWindows()

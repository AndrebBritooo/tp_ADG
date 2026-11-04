#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core05_Conv.py   (na pasta do PL05.zip)
# Convolução com cv.filter2D.

import cv2 as cv
import numpy as np
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Load the image
img = cv.imread(str(PASTA / "imgs" / "astronaut.png"))

# Define the kernel
kernel = np.ones((5, 5), np.float32) / 25

# Apply the kernel to the image
# ddepth = -1 means the output image has the same depth as the source
output = cv.filter2D(img, -1, kernel)

# Display the original and the convolved image
cv.imshow("Original Image", img)
cv.imshow("Convolved Image", output)
cv.waitKey(0)
cv.destroyAllWindows()

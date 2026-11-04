#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core01a_ImgTranslation.py   (na pasta do PL05.zip)
# Translação.

import cv2 as cv
import numpy as np
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Load the image
img = cv.imread(str(PASTA / "imgs" / "astronaut.png"))

# Define the translation matrix
tx = 100
ty = 50
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply the translation
translated_img = cv.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))

# Display the original and translated images
cv.imshow("Original Image", img)
cv.imshow("Translated Image", translated_img)

cv.waitKey(0)
cv.destroyAllWindows()

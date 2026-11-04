#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core01b_ImgRotation.py   (na pasta do PL05.zip)
# Rotação.

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Load the image
img = cv.imread(str(PASTA / "imgs" / "astronaut.png"))

# Get the number of rows and columns of the image
rows, cols = img.shape[:2]

# Define the rotation matrix
# cv.getRotationMatrix2D(center, angle, scale)
# center: center of the rotation in the source image
# angle: rotation angle in degrees
# scale: isotropic scale factor
rot_mat = cv.getRotationMatrix2D((0, 0), 45, 0.5)

# Apply the rotation to the image
rot_img = cv.warpAffine(img, rot_mat, (cols, rows))

# Define the rotation matrix with the center of the image as the center of rotation
rot_mat2 = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)

# Apply the rotation to the image
rot_img2 = cv.warpAffine(img, rot_mat2, (cols, rows))

# Display the rotated images
cv.imshow("rot_img", rot_img)
cv.imshow("rot_img2", rot_img2)

# Wait for a key press to exit
cv.waitKey(0)

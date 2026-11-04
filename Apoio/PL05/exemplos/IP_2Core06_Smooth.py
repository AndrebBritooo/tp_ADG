#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core06_Smooth.py   (na pasta do PL05.zip)
# Média, Gaussiano, mediana e bilateral.

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img = cv.imread(str(PASTA / "imgs" / "astronaut.png"))
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

average = cv.blur(img, ksize=(3, 3))  # Average blur over a NxM mask
gauss = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=2)  # Gaussian blur: NxM mask; sigma; border type
median = cv.medianBlur(img, ksize=5)  # Median filter, NxN mask, good for salt-and-pepper noise
bilateral = cv.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)  # Edge preserving filter

cv.imshow("Original", img)
cv.imshow("Average Blur", average)
cv.imshow("Gaussian Blur", gauss)
cv.imshow("Median Blur", median)
cv.imshow("Bilateral Filter", bilateral)

cv.waitKey(0)

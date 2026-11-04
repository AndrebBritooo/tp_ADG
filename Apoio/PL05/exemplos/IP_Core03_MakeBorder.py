#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_Core03_MakeBorder.py   (na pasta do PL05.zip)
# Margens à volta da imagem, com os cinco modos do cv.copyMakeBorder.

import cv2
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

BLUE = [255,0,0]

img1 = cv2.imread(str(PASTA / "imgs" / "astronaut.png"))

cv2.imshow('Ori',img1)

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

cv2.imshow('replicate',replicate)
cv2.imshow('reflect',reflect)
cv2.imshow('reflect101',reflect101)
cv2.imshow('wrap',wrap)
cv2.imshow('constant',constant)

cv2.waitKey(0)

cv2.destroyAllWindows()

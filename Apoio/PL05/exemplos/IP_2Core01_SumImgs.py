#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core01_SumImgs.py   (na pasta do PL05.zip)
# Somar duas imagens com pesos (cv.addWeighted).

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img1 = cv.imread(str(PASTA / "imgs" / "astronaut.png"))
img2 = cv.imread(str(PASTA / "imgs" / "jewelry.png"))

alpha = 0.3
beta = 1 - alpha
gamma = 0

img_out = cv.addWeighted(img1, alpha, img2, beta, gamma)

cv.imshow("Output", img_out)
cv.waitKey(0)
cv.destroyAllWindows()

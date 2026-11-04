#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core09_Canny.py   (na pasta do PL05.zip)
# Contornos com o cv.Canny.

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img = cv.imread(str(PASTA / "imgs" / "astronaut.png"))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(img, threshold1=50, threshold2=250)

# Side by side comparison
compare = cv.hconcat([img, edges])
# OR compare = np.concatenate((img, edges), axis=1)

cv.imshow("Comparison", compare)
cv.waitKey(0)

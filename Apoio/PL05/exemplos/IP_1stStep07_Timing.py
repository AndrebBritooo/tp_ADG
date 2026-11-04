#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep07_Timing.py   (na pasta do PL05.zip)
# Medir quanto tempo demora uma operação (cv.getTickCount).

import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img_orig = cv.imread(str(PASTA / "imgs" / "astronaut.png"), cv.IMREAD_COLOR)
cv.imshow("Original", img_orig)

# cv.getTickCount function returns the number of clock-cycles after a reference event
e1 = cv.getTickCount()

for i in range(3, 100, 2):
    # Blur the image with a filter size of 3, 5, 7, 9, …, 99
    img_blur = cv.medianBlur(img_orig, i)
    print(i)
    cv.imshow("Blurred", img_blur)
    cv.waitKey(100)

e2 = cv.getTickCount()
# cv.getTickFrequency function returns the frequency of clock-cycles,
# or the number of clock-cycles per second
t = (e2 - e1) / cv.getTickFrequency()
print(f"Seconds elapsed: {t:0.2f}")

cv.waitKey(0)
cv.destroyAllWindows()

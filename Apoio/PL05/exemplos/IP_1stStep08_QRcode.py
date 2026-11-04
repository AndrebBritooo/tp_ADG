#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep08_QRcode.py   (na pasta do PL05.zip)
# Ler um QR code; o imgs/QR2.png traz o próprio programa lá dentro.

import cv2 as cv
import numpy as np
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Read image in Grayscale format
img_in = cv.imread(str(PASTA / "imgs" / "QR1.png"), cv.IMREAD_GRAYSCALE)

qr = cv.QRCodeDetector()

# Detect and decode the QR code
data, bbox, img_rectified = qr.detectAndDecode(img_in)

if len(data) > 0:
    print(f"Decoded Data:\n{data}")
    img_rectified = np.uint8(img_rectified)
    cv.imshow("Rectified QR Code", img_rectified)
else:
    print("QR Code not detected")

cv.imshow("Original QR Code", img_in)

cv.waitKey(0)
cv.destroyAllWindows()

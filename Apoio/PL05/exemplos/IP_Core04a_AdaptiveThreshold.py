#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_Core04a_AdaptiveThreshold.py   (na pasta do PL05.zip)
# Limiar global e adaptativo, sem e com desfoque.

import cv2
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img = cv2.imread(str(PASTA / "imgs" / "astronaut.png"),cv2.IMREAD_GRAYSCALE)
height, width = img.shape[:2]
img2 = cv2.medianBlur(img,7)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
ret,th4 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
th5 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th6 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Blur', 'Global Thresholding (v = 127)','Adaptive Mean Thresholding',
          'Adaptive Gaussian Thresholding','Global Thresholding (v = 127) + Blur Applied',
          'Adaptive Mean Thresholding + Blur Applied', 'Adaptive Gaussian Thresholding + Blur Applied']
images = [img, img2, th1, th2, th3, th4, th5, th6]
for i in range(8):
    images[i] = cv2.resize(images[i],(int(width/2),int(height/2)),interpolation = cv2.INTER_CUBIC)
    cv2.imshow(titles[i],images[i])
cv2.waitKey(0)

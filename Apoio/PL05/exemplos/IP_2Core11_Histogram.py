#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6", "matplotlib"]
# ///
# Correr com: uv run exemplos/IP_2Core11_Histogram.py   (na pasta do PL05.zip)
# O histograma de uma região, com uma máscara.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent


img = cv2.imread(str(PASTA / "imgs" / "astronaut.png"))
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(imgray,imgray,mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([imgray],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([imgray],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(imgray, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()

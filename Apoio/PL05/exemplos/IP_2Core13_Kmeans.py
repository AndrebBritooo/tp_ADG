#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core13_Kmeans.py   (na pasta do PL05.zip)
# Segmentação por cor com k-means.

import cv2 as cv
import numpy as np
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

# Load the image
image = cv.imread(str(PASTA / "imgs" / "jewelry.png"))
assert image is not None, "Image not found"

# Reshape the image into a 2D array of pixels, where each pixel has 3 color channels (BGR)
pixel_array = image.reshape((-1, 3))  # Mx3 size (pixels x color channels)

# Convert the pixel values to float32, which is required for k-means clustering
pixel_array = np.float32(pixel_array)

# Define the termination criteria for k-means clustering
criteria = (  # (type, max_iter, epsilon)
    cv.TERM_CRITERIA_EPS  # EPS: accuracy
    + cv.TERM_CRITERIA_MAX_ITER,  # MAX_ITER: max iterations
    10,  # Max number of iterations
    1.0,  # Required accuracy
)

# Specify the number of clusters to group the pixels into
num_clusters = 5

# Perform k-means clustering on the pixel values
compactness, labels, centers = cv.kmeans(
    pixel_array,
    K=num_clusters,
    bestLabels=None,
    criteria=criteria,
    attempts=10,
    flags=cv.KMEANS_PP_CENTERS,
)

# Convert the center values back to uint8, which is the format for image pixel values
centers = np.uint8(centers)

# Replace each pixel's color with the color of the cluster it belongs to
clustered_pixels = centers[labels.flatten()]

# Reshape the 2D array of pixels back into the original image shape
clustered_image = clustered_pixels.reshape((image.shape))

# Display the original image and the clustered image side by side
comparison = cv.hconcat([image, clustered_image])
cv.imshow("Comparison", comparison)
cv.waitKey(0)

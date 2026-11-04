#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_2Core03_DetectColor.py   (na pasta do PL05.zip)
# Deteção do azul em HSV, com a câmara.

import cv2 as cv
import numpy as np

# Create a VideoCapture object to read from the camera
cap = cv.VideoCapture(0)  # Local camera
# cap = cv.VideoCapture("http://<IP do telemóvel>:4747/video")  # IP (remote) camera

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Define the range of blue colors in HSV
# In OpenCV, the H range for HSV is [0, 179]
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Threshold the HSV image to only get blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND the mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)

    # Display the original frame, the blue threshold (mask), and the blue video
    cv.imshow("Original Frame", frame)
    cv.imshow("Blue Threshold (Mask)", mask)
    cv.imshow("Blue Video", res)

    # Wait for the user to press the ESC key to exit
    if cv.waitKey(1) == 27:
        break

# Release the VideoCapture object and close all windows
cap.release()
cv.destroyAllWindows()

#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_UI01_DrawingFuncs.py   (na pasta do PL05.zip)
# Funções de desenho.

import numpy as np
import cv2 as cv

# Initialize a black image of size 512x512 with 3 color channels
image = np.zeros((512, 512, 3), np.uint8)

# Draw a blue line from the top-left to the bottom-right corner, 5px thick
cv.line(image, (0, 0), (511, 511), (255, 0, 0), 5)

# Draw a green rectangle. The top-left corner is at (384, 0)
# The bottom-right corner is at (510, 128), 3px thick
cv.rectangle(image, (384, 0), (510, 128), (0, 255, 0), 3)

# Draw a red circle. The center is at (447, 63), radius 63px, filled (-1)
cv.circle(image, (447, 63), 63, (0, 0, 255), -1)

# Draw a half ellipse. The center is at (256, 256)
# The major and minor axes are 100 and 50 pixels respectively
# The ellipse is drawn from 0 to 180 degrees, filled (-1)
cv.ellipse(image, (256, 256), (100, 50), 0, 0, 180, (255, 255, 0), -1)

# Draw a polygon. The vertices are defined in an array
# The array is reshaped to ROWSx1x2, where ROWS is the number of vertices
# The polygon is closed (True) and drawn in yellow color
vertices = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
#vertices = vertices.reshape((-1, 1, 2))
cv.polylines(image, [vertices], True, (0, 255, 255))

# Add text to the image. The text "OpenCV" is placed at the bottom-left
# The font used is FONT_HERSHEY_SIMPLEX with a scale of 4
# The color of the text is white and the line type is LINE_AA
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(image, "OpenCV", (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

# Display the image
cv.imshow("Draw", image)

# Wait for a key press and then close the window
cv.waitKey(0)
cv.destroyAllWindows()

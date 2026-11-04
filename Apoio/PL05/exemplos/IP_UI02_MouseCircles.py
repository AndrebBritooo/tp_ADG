#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_UI02_MouseCircles.py   (na pasta do PL05.zip)
# Círculos com duplo clique, esquerdo azul e direito vermelho.

import time

import cv2 as cv
import numpy as np

# Constants
WINDOW_NAME = "Image"
CIRCLE_RADIUS = 100
COLOR_BLUE = (255, 0, 0)
COLOR_RED = (0, 0, 255)
EXIT_KEY = 27  # ESC key

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)


# No macOS, as janelas do OpenCV não dão duplo clique: cada um chega como dois cliques simples. Aqui, dois
# cliques no mesmo botão com menos de 0,4 s de intervalo contam como duplo clique (em Windows e Linux, o
# duplo clique do OpenCV continua a chegar como antes).
DOUBLE_CLICK_TIME = 0.4  # seconds
last_click = {}
as_double = {cv.EVENT_LBUTTONDOWN: cv.EVENT_LBUTTONDBLCLK, cv.EVENT_RBUTTONDOWN: cv.EVENT_RBUTTONDBLCLK}


# Function to handle mouse events
def handle_mouse_event(event, row, col, flags, param):
    if event in as_double:  # a single click: a double click if the previous one was just before
        now = time.monotonic()
        is_double = now - last_click.get(event, -1.0) < DOUBLE_CLICK_TIME
        last_click[event] = -1.0 if is_double else now
        if not is_double:
            return
        event = as_double[event]
    if event == cv.EVENT_LBUTTONDBLCLK:  # double click left button
        cv.circle(img, (row, col), CIRCLE_RADIUS, COLOR_BLUE, -1)
    if event == cv.EVENT_RBUTTONDBLCLK:  # double click right button
        cv.circle(img, (row, col), CIRCLE_RADIUS, COLOR_RED, -1)


# Create a window and bind the mouse event handler to it
cv.namedWindow(WINDOW_NAME)
cv.setMouseCallback(WINDOW_NAME, handle_mouse_event)

# Main loop
while True:
    cv.imshow(WINDOW_NAME, img)
    if cv.waitKey(1) == EXIT_KEY:
        break

# Clean up
cv.destroyAllWindows()

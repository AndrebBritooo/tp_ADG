#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_UI03_Mouse.py   (na pasta do PL05.zip)
# Desenhar com o rato; a tecla t troca retângulo e círculo.

import cv2 as cv
import numpy as np


class MouseDraw:
    # Initialize class constants
    WINDOW_SIZE = (512, 512, 3)  # OpenCV window size
    RECTANGLE_COLOR = (0, 255, 0)
    CIRCLE_COLOR = (0, 0, 255)
    SWITCH_SHAPE_KEY = ord("t")  # key to switch between rectangle and circle
    EXIT_KEY = 27  # ESC key

    def __init__(self):
        # Initialize variables
        self.is_drawing = False  # true if LMB is pressed
        self.is_rectangle = True  # draw rectangle on True; toggle with 't'
        self.start_row, self.start_col = -1, -1  # initial position of mouse

        # Create a black image
        self.img = np.zeros(self.WINDOW_SIZE, np.uint8)

    # Mouse callback function
    def draw_shape(self, event, row, col, flags, param):
        """
        This function is called every time the mouse is clicked or moved.
        Draws a rectangle or a circle depending on the value of self.is_rectangle.
        """
        match event:
            case cv.EVENT_LBUTTONDOWN:
                self.is_drawing = True
                self.start_row, self.start_col = row, col

            case cv.EVENT_MOUSEMOVE:
                if self.is_drawing:
                    if self.is_rectangle:
                        cv.rectangle(
                            self.img,
                            (self.start_row, self.start_col),
                            (row, col),
                            self.RECTANGLE_COLOR,
                            -1,
                        )
                    else:
                        cv.circle(self.img, (row, col), 5, self.CIRCLE_COLOR, -1)

            case cv.EVENT_LBUTTONUP:
                self.is_drawing = False

    def run(self):
        """
        This function creates a window and binds the mouse callback function to it.
        Loops until the user presses the ESC key.
        """
        # Create a window and bind the function to window
        cv.namedWindow("Drawing")
        cv.setMouseCallback("Drawing", self.draw_shape)

        # Loop until escape key is pressed
        while True:
            cv.imshow("Drawing", self.img)

            key = cv.waitKey(1)
            match key:
                case self.SWITCH_SHAPE_KEY:
                    # Toggle between rectangle and circle
                    self.is_rectangle = not self.is_rectangle
                case self.EXIT_KEY:
                    break

        # Close all windows
        cv.destroyAllWindows()


if __name__ == "__main__":
    draw = MouseDraw()
    draw.run()

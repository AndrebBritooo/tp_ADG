#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6"]
# ///
# Correr com: uv run exemplos/IP_1stStep06_ReadSavePixels.py   (na pasta do PL05.zip)
# Ler e mudar píxeis; a forma da imagem, com a altura primeiro.

#%% Initialization
import cv2 as cv
from pathlib import Path

# A pasta do zip (a que tem exemplos/, imgs/ e videos/), a partir deste ficheiro: o programa encontra as
# imagens corra de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent

img_color = cv.imread(str(PASTA / "imgs" / "astronaut.png"), cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

altura, largura = img_gray.shape[:2]  # (linhas, colunas): a altura vem primeiro
print(altura, largura)

# Define position (a,b) to scan pixel value
a = 100
b = 150

#%% ----- PART1 -----
print("\nPart 1")
# Print the gray at coordinate (a,b) in the image
px_gray = img_gray[a, b]

# Print the BGR color at coordinate (a,b) in the image
px = img_color[a, b]
print("BGR ", px)

# Accessing only the Blue pixel
px_blue = img_color[a, b, 0]  # 0: Blue channel
print("Blue ", px_blue)

# Accessing only the Green pixel
px_green = img_color[a, b, 1]  # 1: Green channel
print("Green ", px_green)

# Accessing only the Red pixel
px_red = img_color[a, b, 2]  # 2: Red channel
print("Red ", px_red)

#%% ----- PART2 -----
print("\nPart 2")
# Change color to white
img_color[a, b] = [255, 255, 255]
print(img_color[a, b])

# Accessing Red value at pos (a,b)
print(img_color[a, b, 2])

# Modifying Red value
img_color[a, b, 2] = 100
print(img_color[a, b, 2])

#%% ----- PART3 -----
print("\nPart 3")
# Returns a tuple of number of rows, columns and channels: (512, 512, 3)
altura, largura, canais = img_color.shape
print(img_color.shape)
print(altura, largura, canais)

# Grayscale, tuple returned contains only number of rows and columns
altura_g, largura_g = img_gray.shape
print(img_gray.shape)
print(altura_g, largura_g)

# Total number of pixels is accessed by img.size
print(img_color.size)

# Image datatype is obtained by img.dtype
print(img_color.dtype)

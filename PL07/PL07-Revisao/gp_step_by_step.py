#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "opencv-python>=5.0,<6",
# ]
# ///
"""Os passos da deteção de movimento, todos ao mesmo tempo, numa grelha de 3 x 3.

    uv run gp_step_by_step.py

Cada quadrado é um passo, do fotograma da câmara às células da imagem onde houve movimento:

0. camera      o fotograma como vem da câmara;
1. gray        cvtColor: um só canal, o brilho;
2. blur        GaussianBlur: menos ruído da câmara;
3. difference  absdiff entre o fotograma anterior e o atual; com a barra «MOG2» a 1, o primeiro plano do MOG2;
4. threshold   o que passa o limiar fica branco;
5. dilate      as manchas engrossam e juntam-se;
6. boxes       os contornos e as caixas; a vermelho, as pequenas de mais, que se deitam fora;
7. cells       a imagem partida numa grelha de células quase quadradas, cada uma com a sua fração de branco;
8. grid        as células com movimento pintadas por cima da imagem, mais opacas quanto mais movimento têm.

As barras mudam os valores com o programa a correr: «threshold» (o limiar), «blur» (o tamanho do núcleo),
«dilate» (as vezes que se dilata), «cell %» (o branco a partir do qual uma célula se pinta) e «MOG2» (0: diferença
de fotogramas; 1: subtração de fundo). Teclas: espaço congela a imagem (as barras continuam a mexer no fotograma
congelado); q sai.
"""
import sys

import cv2 as cv
import numpy as np

# --- o que se escolhe antes de correr -------------------------------------------------------------------------
CAMERA = 0                         # o índice da câmara: 0 é a do portátil
CAMERA_SIZE = (640, 480)           # o tamanho pedido à câmara; ela pode dar outro, e o programa adapta-se
GRID_COLUMNS = 20                  # as colunas; as linhas seguem a proporção da imagem (20 x 15 a 640 x 480)
COLOR = (0, 0, 255)                # a cor das células com movimento (BGR: vermelho)
MAX_OPACITY = 0.7                  # a opacidade de uma célula toda branca; com menos movimento, mais transparente
MIN_AREA = 0.0016                  # as caixas menores do que esta fração da imagem são ruído

# os valores com que as barras começam, e o máximo de cada uma
START = {"threshold": 25, "blur": 10, "dilate": 2, "cell %": 10, "MOG2": 0}
MAXIMUM = {"threshold": 100, "blur": 20, "dilate": 6, "cell %": 100, "MOG2": 1}

WINDOW = "group project, step by step"
STEPS = ["camera", "gray", "blur", "difference", "threshold", "dilate", "boxes", "cells", "grid"]
FONT = cv.FONT_HERSHEY_SIMPLEX
YELLOW, RED, BLUE, GRAY, WHITE = (0, 255, 255), (0, 0, 255), (255, 128, 0), (90, 90, 90), (255, 255, 255)
KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))   # o elemento da abertura, no MOG2


# --- os passos ------------------------------------------------------------------------------------------------
def detection_steps(frame, previous_gray, mog2, values, learning_rate):
    """Os passos 0 a 5, cada um como imagem: da câmara até à máscara final."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)            # passo 1: um só canal
    k = 2 * values["blur"] + 1                              # o núcleo tem de ser ímpar: 1, 3, 5, ... 21, ...
    blur = cv.GaussianBlur(gray, (k, k), 0)                 # passo 2: menos ruído
    if mog2 is None:                                        # diferença de fotogramas
        previous_blur = cv.GaussianBlur(previous_gray, (k, k), 0)
        difference = cv.absdiff(previous_blur, blur)        # passo 3: o que mudou desde o fotograma anterior
        _, binary = cv.threshold(difference, values["threshold"], 255, cv.THRESH_BINARY)   # passo 4
    else:                                                   # subtração de fundo
        difference = mog2.apply(frame, learningRate=learning_rate)   # passo 3: 255 mexeu, 127 sombra, 0 fundo
        _, binary = cv.threshold(difference, 200, 255, cv.THRESH_BINARY)   # passo 4: as sombras ficam de fora
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, KERNEL)            # a abertura apaga os pontos soltos
    mask = cv.dilate(binary, None, iterations=values["dilate"])   # passo 5: as manchas engrossam e juntam-se
    return {"camera": frame, "gray": gray, "blur": blur, "difference": difference, "threshold": binary,
            "dilate": mask}


def draw_boxes(frame, mask, min_area):
    """Passo 6: os contornos a azul, as caixas que ficam a amarelo e as que se deitam fora a vermelho."""
    image = frame.copy()
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, BLUE, 2)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        big = cv.contourArea(c) >= min_area
        cv.rectangle(image, (x, y), (x + w, y + h), YELLOW if big else RED, 2)
    return image


def grid_size(width, height):
    """As colunas e as linhas da grelha: as linhas na proporção da imagem, para as células serem quase quadradas."""
    return GRID_COLUMNS, max(1, round(GRID_COLUMNS * height / width))


def cell_motion(mask, columns, rows):
    """Passo 7: a fração de branco de cada célula, de 0.0 a 1.0, numa matriz de rows x columns.

    A máscara passa de 0 e 255 a 0.0 e 1.0, e o cv.resize com INTER_AREA faz a média dos píxeis de cada célula,
    que é a fração de branco."""
    return cv.resize(mask / 255.0, (columns, rows), interpolation=cv.INTER_AREA)


def draw_cells(motion, width, height):
    """O passo 7 como imagem: cada célula com o cinzento da sua fração de branco, e as linhas da grelha."""
    rows, columns = motion.shape
    image = cv.resize((motion * 255).astype(np.uint8), (width, height), interpolation=cv.INTER_NEAREST)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for c in range(1, columns):
        x = c * width // columns
        cv.line(image, (x, 0), (x, height - 1), GRAY, 1)
    for r in range(1, rows):
        y = r * height // rows
        cv.line(image, (0, y), (width - 1, y), GRAY, 1)
    return image


def draw_grid(frame, motion, percent):
    """Passo 8: as células acima de `percent` % de branco pintadas com COLOR por cima da imagem; a opacidade é
    proporcional à fração de branco da célula, até MAX_OPACITY numa célula toda branca."""
    alpha = np.where(motion * 100 > percent, MAX_OPACITY * motion, 0.0)   # as que ficam abaixo: transparentes
    height, width = frame.shape[:2]
    alpha = cv.resize(alpha, (width, height), interpolation=cv.INTER_NEAREST)[:, :, None]   # célula a célula
    color = np.full_like(frame, COLOR)
    return (alpha * color + (1 - alpha) * frame).astype(np.uint8)


# --- o que a janela mostra ------------------------------------------------------------------------------------
def to_bgr(image):
    """As imagens de um canal (cinzento, máscaras) passam a três, para se poderem juntar e escrever a cores."""
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR) if image.ndim == 2 else image


def mosaic(images, width, height, paused):
    """Os nove passos numa grelha de 3 x 3, cada um com o seu número e nome, e as teclas numa faixa por cima."""
    w, h = width // 3, height // 3
    tiles = []
    for n, name in enumerate(STEPS):
        tile = cv.resize(to_bgr(images[name]), (w, h))
        cv.putText(tile, f"{n} {name}", (6, 20), FONT, 0.55, YELLOW, 2)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 3]) for i in (0, 3, 6)]
    grid = cv.resize(np.vstack(rows), (width, height))     # do tamanho do fotograma
    bar = np.zeros((56, width, 3), np.uint8)
    cv.putText(bar, "all steps" + ("   (frozen)" if paused else ""), (10, 22), FONT, 0.65, WHITE, 2)
    cv.putText(bar, "space: freeze   q: quit", (10, 46), FONT, 0.5, WHITE, 1)
    return np.vstack([bar, grid])


# --- a aplicação ----------------------------------------------------------------------------------------------
def main():
    # 1. a câmara: a resolução é a do primeiro fotograma, e a grelha calcula-se a partir dela
    camera = cv.VideoCapture(CAMERA)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
    ok, frame = camera.read()
    if not ok:
        sys.exit(f"Camera {CAMERA} gave no image. Try another index in CAMERA.")
    height, width = frame.shape[:2]
    columns, rows = grid_size(width, height)
    min_area = int(MIN_AREA * width * height)

    # 2. a janela e as barras
    cv.namedWindow(WINDOW)
    for name, value in START.items():
        cv.createTrackbar(name, WINDOW, value, MAXIMUM[name], lambda v: None)

    # 3. o estado: o fotograma anterior e o MOG2 (só quando a barra está a 1)
    previous_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mog2 = None
    paused = False

    # 4. o ciclo: os nove passos em cada fotograma, juntos numa só imagem
    while ok:
        values = {name: cv.getTrackbarPos(name, WINDOW) for name in START}
        if values["MOG2"] and mog2 is None:            # o MOG2 começa a aprender o fundo quando se liga
            mog2 = cv.createBackgroundSubtractorMOG2(detectShadows=True)
        if not values["MOG2"]:
            mog2 = None
        images = detection_steps(frame, previous_gray, mog2, values, 0 if paused else 0.001)
        images["boxes"] = draw_boxes(frame, images["dilate"], min_area)
        motion = cell_motion(images["dilate"], columns, rows)
        images["cells"] = draw_cells(motion, width, height)
        images["grid"] = draw_grid(frame, motion, values["cell %"])
        if mog2 is None:                               # a diferença é escura: x4 para se ver
            images["difference"] = cv.convertScaleAbs(images["difference"], alpha=4)
        cv.imshow(WINDOW, mosaic(images, width, height, paused))

        key = cv.waitKey(30 if paused else 1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):                            # congela ou descongela a imagem
            paused = not paused
        if not paused:                                 # o atual passa a anterior, e lê-se o seguinte
            previous_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ok, frame = camera.read()

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

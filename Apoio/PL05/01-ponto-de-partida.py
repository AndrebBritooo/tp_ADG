#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "opencv-python>=5.0,<6",
# ]
# ///
"""PL05, secção 1: o ponto de partida, que é o programa final da PL04.

    uv run 01-ponto-de-partida.py [pasta] [indice da camara]

É o programa com que acaba a secção 8 do guião PL04, sem uma vírgula mudada: deteção de movimento por diferença de
fotogramas em toda a imagem, e uma animação que corre enquanto houver movimento. Esta aula parte daqui e
acrescenta as **zonas**: onde é que o movimento aconteceu, e qual é o objeto que responde.
"""
import pathlib
import sys

import cv2 as cv

pasta = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets/zona1_cadeira")
indice = int(sys.argv[2]) if len(sys.argv) > 2 else 0

LADO, X, Y = 160, 400, 250
LIMIAR, AREA_MINIMA = 25, 500
PARAGEM = 15          # fotogramas sem movimento antes de a animação parar

fotogramas = []
for ficheiro in sorted(pasta.glob("*.png")):
    objeto = cv.imread(str(ficheiro), cv.IMREAD_UNCHANGED)
    if objeto is None:
        sys.exit(f"Não consegui ler {ficheiro.name}. O caminho está certo?")
    if objeto.ndim != 3 or objeto.shape[2] != 4:
        sys.exit(f"{ficheiro.name} não tem canal alfa (transparência). No "
                 "Blender: Render Properties › Film › Transparent, e no "
                 "Output, PNG com RGBA.")
    fotogramas.append(cv.resize(objeto, (LADO, LADO)))
if not fotogramas:
    sys.exit(f"Não há PNG em {pasta}.")

camara = cv.VideoCapture(indice)
if not camara.isOpened():
    sys.exit(f"Não consegui abrir a câmara {indice}.")

anterior, i, quietos = None, 0, PARAGEM
while True:
    ok, fotograma = camara.read()
    if not ok:
        break

    cinzento = cv.cvtColor(fotograma, cv.COLOR_BGR2GRAY)
    suave = cv.GaussianBlur(cinzento, (21, 21), 0)
    houve_movimento = False
    if anterior is not None:
        diferenca = cv.absdiff(anterior, suave)
        _, limiar = cv.threshold(diferenca, LIMIAR, 255, cv.THRESH_BINARY)
        dilatado = cv.dilate(limiar, None, iterations=2)
        contornos, _ = cv.findContours(dilatado, cv.RETR_EXTERNAL,
                                       cv.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            if cv.contourArea(contorno) < AREA_MINIMA:
                continue
            houve_movimento = True
            x, y, w, h = cv.boundingRect(contorno)
            cv.rectangle(fotograma, (x, y), (x + w, y + h), (0, 255, 0), 2)
    anterior = suave

    quietos = 0 if houve_movimento else quietos + 1
    if quietos < PARAGEM:            # ainda a animar
        objeto = fotogramas[i % len(fotogramas)]
        bgr, alfa = objeto[:, :, :3], objeto[:, :, 3] / 255.0
        alfa = alfa[:, :, None]
        zona = fotograma[Y:Y + LADO, X:X + LADO]
        if zona.shape[:2] == (LADO, LADO):
            misturado = alfa * bgr + (1 - alfa) * zona
            fotograma[Y:Y + LADO, X:X + LADO] = misturado.astype("uint8")
        i += 1
    else:
        i = 0                        # parada: a sequência volta ao princípio

    estado = "movimento" if houve_movimento else "parado"
    cv.putText(fotograma, estado, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               0.9, (0, 255, 0), 2)
    cv.imshow("movimento e animacao", fotograma)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv.destroyAllWindows()

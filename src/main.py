#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "opencv-python>=5.0,<6",
#   "rich",
# ]
# ///
from datetime import datetime
from rich import console

import json
import pathlib
import sys

import cv2 as cv

raiz = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets")
indice = int(sys.argv[2]) if len(sys.argv) > 2 else 0

CONFIG = pathlib.Path("config.json")
OMISSAO = {"limiar": 25, "percentagem": 2, "paragem": 15,
           "zonas": [[0, 0, 213, 480], [213, 0, 213, 480],
                     [426, 0, 214, 480]]}
LADO = 160
AREA_MINIMA = 500              # só para desenhar as caixas
VERDE, AMARELO = (0, 255, 0), (0, 255, 255)
JANELA = "afinar"
consola = console.Console(log_path=False)



def carrega():
    """A configuração do ficheiro, com os valores por omissão para o que lá não estiver."""
    valores = dict(OMISSAO)
    if CONFIG.exists():
        try:
            valores.update(json.loads(CONFIG.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            sys.exit(f"{CONFIG} não é um JSON válido ({e}). Apaguem-no para "
                     "recomeçar com os valores por omissão.")
    for n, roi in enumerate(valores["zonas"], start=1):
        if len(roi) != 4:
            sys.exit(f"{CONFIG}: a zona {n} tem de ser [x, y, largura, altura], "
                     f"e está {roi}.")
    return valores


def le_sequencia(pasta):
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
        sys.exit(f"Não há PNG em {pasta}. A pasta desta zona está vazia?")
    return fotogramas


def confere(zonas, largura, altura):
    """Pára com uma mensagem se alguma zona sair da imagem."""
    for n, (x, y, w, h) in enumerate(zonas, start=1):
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > largura or y + h > altura:
            sys.exit(f"A zona {n} {(x, y, w, h)} sai da imagem de {largura} x {altura}. "
                     f"Corrijam as zonas no {CONFIG}.")

def toca(caixa, zona):
    """Verifica se uma caixa de movimento toca uma zona. A caixa é (x, y, w, h) e a zona é (x, y, w, h)."""
    x, y, w, h = caixa
    zx, zy, zw, zh = zona

    return(
        x < zx + zw 
        and zx < x + w
        and y < zy + zh
        and zy < y + h
    )

def comeca(z, caixas):
    """inicia um evento de movimento."""
    inicio = datetime.now()

    evento = {
        "inicio": inicio,
        "caixas": caixas
    }

    consola.log(f"[green]Zona {z + 1} iniciou movimento às {inicio.strftime('%H:%M:%S')} com {len(caixas)} caixas.")
    return evento

def acaba(z, evento):
    """Termina um evento de movimento."""
    duracao = (
        datetime.now() - evento["inicio"]
    ).total_seconds()

    consola.log(
        f"[yellow]zona {z + 1}[/]: acabou, {duracao:.1f} s"
    )


valores = carrega()
zonas = [tuple(int(v) for v in roi) for roi in valores["zonas"]]
pastas = sorted(p for p in raiz.glob("zona*") if p.is_dir())
if not pastas:
    sys.exit(f"Não há pastas de zonas em {raiz} (esperava assets/zona1_..., "
             "assets/zona2_..., uma por zona).")
if len(pastas) != len(zonas):
    sys.exit(f"Há {len(zonas)} zonas no {CONFIG} e {len(pastas)} pastas em "
             f"{raiz}: tem de haver uma pasta por zona.")
sequencias = [le_sequencia(p) for p in pastas]
consola.log("configuração:", valores)

cv.namedWindow(JANELA)
cv.createTrackbar("limiar", JANELA, valores["limiar"], 100, lambda v: None)
cv.createTrackbar("zona %", JANELA, valores["percentagem"], 50, lambda v: None)
cv.createTrackbar("paragem", JANELA, valores["paragem"], 60, lambda v: None)

camara = cv.VideoCapture(indice)
if not camara.isOpened():
    sys.exit(f"Não consegui abrir a câmara {indice}.")
camara.set(cv.CAP_PROP_FRAME_WIDTH, 640)           # as zonas estão em píxeis de 640 x 480
camara.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

i = [0] * len(zonas)
quietos = [valores["paragem"]] * len(zonas)
eventos = [None] * len(zonas)
anterior = None
while True:
    ok, fotograma = camara.read()
    if not ok:
        break
    valores["limiar"] = cv.getTrackbarPos("limiar", JANELA)
    valores["percentagem"] = cv.getTrackbarPos("zona %", JANELA)
    valores["paragem"] = max(1, cv.getTrackbarPos("paragem", JANELA))
    altura, largura = fotograma.shape[:2]
    confere(zonas, largura, altura)

    suave = cv.GaussianBlur(cv.cvtColor(fotograma, cv.COLOR_BGR2GRAY), (21, 21), 0)
    acesas, caixas = set(), []
    if anterior is not None:
        _, limiar = cv.threshold(cv.absdiff(anterior, suave), valores["limiar"],
                                 255, cv.THRESH_BINARY)
        mascara = cv.dilate(limiar, None, iterations=2)
        for z, (x, y, w, h) in enumerate(zonas):
            brancos = cv.countNonZero(mascara[y:y + h, x:x + w])
            if brancos > valores["percentagem"] / 100 * w * h:
                acesas.add(z)
        contornos, _ = cv.findContours(mascara, cv.RETR_EXTERNAL,
                                       cv.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            if cv.contourArea(contorno) >= AREA_MINIMA:
                caixas.append(cv.boundingRect(contorno))

        for x, y, w, h in caixas:
            cv.rectangle(
                fotograma,
                (x, y),
                (x + w, y + h),
                AMARELO,
                2
            )
    anterior = suave


    for z, zona in enumerate(zonas):
        x, y, w, h = zona

        quietos[z] = 0 if z in acesas else quietos[z] + 1

        # Iniciar o evento quando a zona deteta movimento
        if z in acesas and eventos[z] is None:
            caixas_zona = [
                caixa for caixa in caixas
                if toca(caixa, zona)
            ]
            eventos[z] = comeca(z, caixas_zona)

        # Terminar o evento após os frames de paragem
        if eventos[z] is not None and quietos[z] >= valores["paragem"]:
            acaba(z, eventos[z])
            eventos[z] = None

        if quietos[z] < valores["paragem"]:
            objeto = sequencias[z][i[z] % len(sequencias[z])]
            ox, oy = x + (w - LADO) // 2, y + (h - LADO) // 2
            fatia = fotograma[oy:oy + LADO, ox:ox + LADO]
            if fatia.shape[:2] == (LADO, LADO):
                alfa = objeto[:, :, 3] / 255.0
                alfa = alfa[:, :, None]
                fotograma[oy:oy + LADO, ox:ox + LADO] = (
                    alfa * objeto[:, :, :3] + (1 - alfa) * fatia).astype("uint8")
            i[z] += 1
        else:
            i[z] = 0
        cv.rectangle(fotograma, (x, y), (x + w - 1, y + h - 1), VERDE,
                     4 if z in acesas else 1)
        cv.putText(fotograma, str(z + 1), (x + 8, y + 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, VERDE, 2)

    cv.putText(fotograma, "g: gravar config.json   q: sair", (10, altura - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, VERDE, 2)
    cv.imshow(JANELA, fotograma)
    tecla = cv.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == ord("g"):
        CONFIG.write_text(json.dumps(valores, indent=2) + "\n", encoding="utf-8")
        consola.log("gravado em", CONFIG, valores)

    for z, evento in enumerate(eventos):
        if evento is not None:
            acaba(z, evento)
camara.release()
cv.destroyAllWindows()

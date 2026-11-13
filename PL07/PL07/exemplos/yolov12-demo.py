#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python>=5.0,<6", "ultralytics>=8.3", "torch>=2.3"]
# ///
# Correr com: uv run exemplos/yolov12-demo.py   (na pasta do PL07.zip)
# Deteção de objetos com YOLO12, na câmara, num vídeo ou numa ligação.

"""
YOLOv12 real-time detection demo using OpenCV VideoCapture.

This script demonstrates how to:
- Load Ultralytics YOLOv12/YOLO12 models and run detection on a video source.
- Read from webcams, IP streams (HTTP/RTSP), or local video files.
- Display predictions with styled bounding boxes and labels.
- Show an on-screen FPS overlay and keep latency low for live sources.

Behavior and tips:
- Live sources (webcams/streams) use a latest-frame reader to avoid latency build-up.
- Local files are read sequentially and paced to the file's FPS for natural playback.
- Default device prefers CUDA when available; half precision is used on CUDA unless disabled.
- If given a bare filename or a .pt URL, the model is downloaded into the zip folder (next to exemplos/).
- On Windows, DirectShow is used for webcam indices for faster, more reliable capture.

Run via `uv` using the PEP-723 header above. Examples:
  uv run exemplos/yolov12-demo.py --camera 0
  uv run exemplos/yolov12-demo.py --camera video.mp4
  uv run exemplos/yolov12-demo.py --camera http://host:port/video
"""
import argparse
import sys
import time
from typing import Tuple, Union
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
import threading
from urllib.error import URLError, HTTPError

import cv2
import numpy as np
from ultralytics import YOLO
import torch

# A pasta do zip (a que tem exemplos/), a partir deste ficheiro: os pesos do modelo ficam lá, corra o
# programa de onde correr, em Windows, macOS ou Linux.
PASTA = Path(__file__).resolve().parent.parent


# --- Mapa do programa --------------------------------------------------------------------------------------------
# O essencial está no main(), no fim do ficheiro, em cinco passos:
#   1. onde corre a rede: na placa gráfica NVIDIA (cuda), se houver, ou no processador (cpu);
#   2. o modelo: YOLO("yolo12n.pt") carrega uma rede já treinada no conjunto COCO, com 80 classes;
#   3. a fonte das imagens: a câmara, um ficheiro de vídeo ou uma ligação;
#   4. o ciclo: ler um fotograma, model.predict(fotograma), desenhar as caixas e mostrar a janela;
#   5. as imagens por segundo, no canto, e as teclas (q ou Esc sai, m espelha).
# As funções até lá são de apoio (argumentos, câmara, descarga dos pesos, cores e desenho): leiam-nas depois.

# Os argumentos da linha de comandos; uv run exemplos/yolov12-demo.py --help mostra-os todos.
def parse_args():
    """Parse command-line arguments for the detection demo.

    Returns:
        argparse.Namespace: Parsed command-line options controlling model, source,
        visualization and performance knobs.
    """
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLOv12 webcam object detection"
    )
    parser.add_argument(
        "--model",
        default="auto",
        help=(
            "Model weights path/name or 'auto' for best available (tries x/l/m/s/n)."
        ),
    )
    parser.add_argument(
        "--camera",
        default="0",
        help=(
            "Video source: index, URL, or file path (e.g. '0', 'video.mp4', 'http://host:port/video', 'rtsp://...')"
        ),
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Computation device: 'auto' (default) chooses CUDA if available, else CPU; "
            "or specify 'cuda', 'cuda:0', or 'cpu'"
        ),
    )
    parser.add_argument(
        "--mirror", action="store_true", help="Mirror the webcam feed horizontally"
    )
    parser.add_argument(
        "--show-fps", action="store_true", default=True, help="Overlay FPS on the video"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference size (pixels), e.g. 640"
    )
    parser.add_argument(
        "--no-half", action="store_true", help="Disable half precision on CUDA"
    )
    return parser.parse_args()


def _is_url(src: str) -> bool:
    """Check whether a string points to a URL.

    Args:
        src (str): Input string to classify.

    Returns:
        bool: True if the string starts with a known URL scheme (e.g. http, rtsp),
        otherwise False.
    """
    s = src.lower()
    return s.startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://"))


def parse_camera_source(src: str) -> Union[int, str]:
    """Interpret the source as an integer index or a string path/URL.

    Args:
        src (str): The value provided by the user via ``--camera``.

    Returns:
        Union[int, str]: Camera index if ``src`` can be parsed as an int, else the
        original string for a file path or URL.
    """
    try:
        return int(src)
    except (TypeError, ValueError):
        # Caller should pass a complete URL when using network cameras.
        return src


def open_capture(camera_source: Union[int, str], width: int, height: int) -> cv2.VideoCapture:
    """Create and configure a ``cv2.VideoCapture`` for the provided source.

    For camera indices on Windows, the DirectShow backend is preferred to reduce
    open latency. For string sources (file path or URL), the function forwards
    the value directly to OpenCV and attempts to lower the internal buffer size
    where supported to reduce latency.

    Args:
        camera_source (Union[int, str]): Source to open (index, file path, or URL).
        width (int): Desired capture width when opening a webcam (ignored for files/URLs).
        height (int): Desired capture height when opening a webcam (ignored for files/URLs).

    Returns:
        cv2.VideoCapture: An opened capture handle ready for reading frames.

    Raises:
        RuntimeError: If the source cannot be opened.
    """
    # Pass the source directly to OpenCV; on Windows prefer DirectShow for local indices
    if isinstance(camera_source, int):
        cap = None
        if sys.platform == "win32":
            try:
                cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
            except Exception:
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {camera_source}")
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    # String source: pass full URL straight through
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source {camera_source}")
    # Try to keep buffer small to reduce latency where supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


# Apoio: lê a câmara numa thread à parte e guarda só o fotograma mais recente. A rede é mais lenta do que a
# câmara; sem isto, as imagens iam-se acumulando e a janela mostrava o que se passou há uns segundos.
class LatestFrameReader:
    """Continuously grab frames in a thread and expose only the freshest.

    Intended for live sources (webcams/streams) to minimize perceived latency by
    dropping stale frames. Prefer sequential reads (without this class) for
    local file playback to preserve natural pacing.

    Args:
        cap (cv2.VideoCapture): An opened capture from which frames are read.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._stopped = False
        self._ready = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        """Start the reader thread.

        Returns:
            LatestFrameReader: ``self`` for call chaining.
        """
        self._t.start()
        return self

    def _loop(self):
        """Background loop that continuously reads and snapshots the last frame."""
        while not self._stopped:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = f
                self._ready.set()

    def read(self):
        """Get the most recent frame.

        Returns:
            Tuple[bool, np.ndarray]: ``(ok, frame)`` where ``ok`` is True when a
            frame is available and ``frame`` is a copy of the latest frame.
        """
        # Wait briefly for first frame
        if not self._ready.is_set():
            self._ready.wait(timeout=1.0)
        with self._lock:
            f = None if self._frame is None else self._frame.copy()
        return f is not None, f

    def isOpened(self) -> bool:
        """Mirror of ``VideoCapture.isOpened`` for convenience.

        Returns:
            bool: True if the underlying capture is open.
        """
        return self.cap.isOpened()

    def stop(self):
        """Signal the thread to stop and join briefly."""
        self._stopped = True
        try:
            self._t.join(timeout=1.0)
        except Exception:
            pass

    def release(self):
        """Stop and release the underlying capture."""
        self.stop()
        try:
            self.cap.release()
        except Exception:
            pass


def _is_basename(path_str: str) -> bool:
    """Check if a string is only a filename.

    Args:
        path_str (str): Path-like string to examine.

    Returns:
        bool: True if ``path_str`` contains no directory components.
    """
    try:
        return Path(path_str).name == path_str
    except Exception:
        return False


def _download_to_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Download a URL to a file atomically (via a temporary ``.part`` file).

    Args:
        url (str): Source URL to download.
        dest (Path): Target file path to write.
        chunk_size (int): Read size in bytes for streaming download.

    Raises:
        URLError: If there is a network-related failure.
        HTTPError: If the HTTP request fails.
        OSError: If the file cannot be written.
    """
    dest_tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url) as r, open(dest_tmp, "wb") as f:
        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    dest_tmp.replace(dest)


# Apoio: os pesos da rede (o ficheiro .pt) ficam na pasta do zip; da primeira vez, descarregam-se da Ultralytics.
def ensure_local_weights(weights: str) -> str:
    """Ensure model weights exist in the zip folder. Returns a path or original string.

    - If `weights` is a path and exists, return it.
    - If it's a URL to a .pt file, download to the zip folder and return the local path.
    - If it's a bare filename like 'yolov12n.pt' and not present, try to download
      from Ultralytics assets 'latest' release into the zip folder, then return the path.
    - Otherwise, return the original string and let Ultralytics handle it.
    """
    try:
        p = Path(weights)
    except Exception:
        p = None

    # Already a local file
    if p is not None and p.exists():
        return str(p)

    # URL to a .pt weights file
    if isinstance(weights, str) and _is_url(weights) and weights.lower().endswith(".pt"):
        local = PASTA / Path(urlparse(weights).path).name
        if not local.exists():
            try:
                print(f"Downloading weights to {local} ...", file=sys.stderr)
                _download_to_file(weights, local)
            except (URLError, HTTPError, OSError) as e:
                print(f"Warning: failed to download weights from URL: {e}", file=sys.stderr)
                return weights  # fall back to remote URL handling
        return str(local)

    # Bare filename like 'yolo12n.pt' or 'yolov12n.pt'
    if isinstance(weights, str) and _is_basename(weights) and weights.lower().endswith(".pt"):
        local = PASTA / weights
        if local.exists():
            return str(local)

        wl = weights.lower()
        url = None
        # Ultralytics official naming (no 'v')
        if wl.startswith("yolo12"):
            url = f"https://github.com/ultralytics/assets/releases/latest/download/{weights}"
        # Authors' repo naming (with 'v')
        elif wl.startswith("yolov12"):
            # Default to turbo release which README labels as default
            url = f"https://github.com/sunsmarterjie/yolov12/releases/download/turbo/{weights}"

        if url is not None:
            try:
                print(f"Attempting to download {weights} to {local} ...", file=sys.stderr)
                _download_to_file(url, local)
                return str(local)
            except (URLError, HTTPError, OSError) as e:
                print(
                    "Warning: could not download weights to the zip folder; "
                    "falling back to Ultralytics cache. Details: " + str(e),
                    file=sys.stderr,
                )
                # fall through to return original

    return weights


# Apoio: o desenho, que já não é YOLO mas OpenCV: uma cor por classe, a etiqueta e a caixa com os cantos marcados.
def class_color(cid: int) -> Tuple[int, int, int]:
    """Generate a deterministic BGR color for a class index.

    Args:
        cid (int): Class index.

    Returns:
        Tuple[int, int, int]: Color as ``(B, G, R)``.
    """
    # Deterministic color per class (BGR)
    rng = np.random.default_rng(seed=cid + 12345)
    color = rng.integers(0, 255, size=3, dtype=np.uint8).tolist()
    return int(color[2]), int(color[1]), int(color[0])


def _class_name(names, cls_idx: int) -> str:
    """Resolve a class name from Ultralytics model names.

    Args:
        names: Model.names (dict or list).
        cls_idx (int): Class index.

    Returns:
        str: Class name or the index as string if unavailable.
    """
    i = int(cls_idx)
    if isinstance(names, dict):
        return str(names.get(i, i))
    if isinstance(names, (list, tuple)):
        return str(names[i]) if 0 <= i < len(names) else str(i)
    return str(i)


def draw_label(img, x, y, label, color, alpha: float = 0.65):
    """Draw a semi-transparent label background with outlined text.

    Args:
        img (np.ndarray): BGR image (modified in-place).
        x (int): Bounding-box left coordinate used as the label anchor.
        y (int): Bounding-box top coordinate used as the label anchor.
        label (str): Text content to render.
        color (Tuple[int, int, int]): Background color in BGR.
        alpha (float): Background opacity in ``[0, 1]``.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 3
    th_full = th + 2 * pad
    # Prefer drawing above the box; if out of frame, draw below
    top = y - th_full
    bottom = y
    if top < 0:
        top = y
        bottom = y + th_full
    left = x
    right = x + tw + 2 * pad
    # Clip within image bounds
    h, w = img.shape[:2]
    left = max(0, left)
    right = min(w - 1, right)
    top = max(0, top)
    bottom = min(h - 1, bottom)

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # White text with subtle black outline for contrast
    tx = left + pad
    ty = bottom - pad - 1
    cv2.putText(img, label, (tx, ty), font, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (tx, ty), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_fancy_box(img, x1, y1, x2, y2, color, thickness: int = 2, length: int = 12, alpha: float = 0.15):
    """Draw a bounding box with a soft fill and corner accents.

    Args:
        img (np.ndarray): BGR image (modified in-place).
        x1 (int): Left coordinate.
        y1 (int): Top coordinate.
        x2 (int): Right coordinate.
        y2 (int): Bottom coordinate.
        color (Tuple[int, int, int]): Box color in BGR.
        thickness (int): Corner segment thickness.
        length (int): Corner segment length in pixels.
        alpha (float): Fill opacity in ``[0, 1]``.
    """
    # Translucent fill inside box
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Outer thin border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Decorative corner accents
    l = max(6, length)
    t = max(2, thickness)
    # Top-left
    cv2.line(img, (x1, y1), (x1 + l, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, t, cv2.LINE_AA)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - l, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, t, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + l, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, t, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - l, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, t, cv2.LINE_AA)


def main():
    """Main application loop for the detection demo.

    Opens the video source, runs YOLO predictions per frame, draws styled results,
    and handles basic keyboard controls (``q``/Esc to quit, ``m`` to mirror).
    """
    args = parse_args()

    # 1. Onde corre a rede: "cuda" é a placa gráfica NVIDIA, muito mais rápida; sem ela, o processador ("cpu").
    # Resolve device preference: default to CUDA when available
    req_device = (args.device or "auto").lower()
    if req_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = req_device

    # Afinações de desempenho para a placa gráfica; sem NVIDIA, não fazem nada e podem ignorar-se.
    # Optimize runtimes where possible
    try:
        torch.backends.cudnn.benchmark = True
        # Prefer new TF32 controls (no torch.set_float32_matmul_precision to avoid warnings)
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.fp32_precision = "tf32"
            except Exception:
                pass
            try:
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            except Exception:
                pass
        cv2.setUseOptimized(True)
    except Exception:
        pass

    # Resolve best model automatically when requested
    def _choose_best_default_weights() -> str:
        prefs = [
            "yolo12x.pt", "yolov12x.pt",
            "yolo12l.pt", "yolov12l.pt",
            "yolo12m.pt", "yolov12m.pt",
            "yolo12s.pt", "yolov12s.pt",
            "yolo12n.pt", "yolov12n.pt",
        ]
        for cand in prefs:  # only the ones already in the zip folder: yolo12x is 119 MB and slow on a CPU
            if (PASTA / cand).exists():
                return str(PASTA / cand)
        # Last resort
        return ensure_local_weights("yolo12n.pt")

    # 2. O modelo: os pesos de uma rede já treinada no COCO (80 classes: person, chair, cup, laptop, ...).
    #    Com --model auto, usa o maior que já estiver na pasta do zip; se não houver nenhum, descarrega o
    #    yolo12n, o mais pequeno e o mais rápido (n, s, m, l, x: do mais pequeno para o maior).
    if (args.model or "auto").lower() in {"auto", "best"}:
        weights_arg = _choose_best_default_weights()
        print(f"Using model: {weights_arg}", file=sys.stderr)
    else:
        weights_arg = ensure_local_weights(args.model)

    try:
        # YOLO(...) carrega a rede a partir do ficheiro dos pesos; .to(...) põe-na na placa gráfica ou no
        # processador.
        model = YOLO(weights_arg).to(resolved_device)
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}", file=sys.stderr)
        print(
            "Tip: Update ultralytics (`uv pip install -U ultralytics`) or try a different weights file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # model.names liga o número de cada classe ao seu nome: 0 é person, 56 é chair, e assim por diante.
    names = model.names if hasattr(model, "names") else {}
    source = parse_camera_source(args.camera)
    # If a URL ends with '/', strip it to avoid some backends failing
    if isinstance(source, str) and _is_url(source) and source.endswith('/'):
        source = source.rstrip('/')
    # 3. A fonte das imagens: --camera 0 (a câmara do portátil), 1 (outra câmara), um vídeo ou uma ligação.
    #    Numa fonte ao vivo, o LatestFrameReader fica só com o fotograma mais recente.
    cap = open_capture(source, args.width, args.height)
    # Use drop-frame reader only for live sources (index or URL). For local files,
    # read sequentially and pace playback to the file's FPS to avoid racing ahead.
    is_file_input = isinstance(source, str) and not _is_url(source)
    reader = None if is_file_input else LatestFrameReader(cap).start()

    window_name = "YOLOv12 Webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # FPS smoothing via an exponential moving average helps stabilize readings.
    fps = 0.0
    t_last = time.perf_counter()
    smoothing = 0.9  # exponential moving average for fps
    # Target pacing for file playback
    target_dt = None
    if is_file_input:
        fps_cap = cap.get(cv2.CAP_PROP_FPS)
        try:
            if fps_cap and fps_cap > 1e-6 and np.isfinite(fps_cap):
                target_dt = 1.0 / float(fps_cap)
        except Exception:
            target_dt = None
        if target_dt is None:
            target_dt = 1.0 / 30.0

    # 4. O ciclo: um fotograma de cada vez, até q ou Esc.
    try:
        while True:
            loop_start = time.perf_counter()
            if reader is not None:
                ok, frame = reader.read()
            else:
                ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from source", file=sys.stderr)
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            # Device/precision for inference
            inference_device = resolved_device
            if req_device == "auto":
                inference_device = "cuda" if torch.cuda.is_available() else "cpu"
            use_half = (not args.no_half) and inference_device.startswith("cuda") and torch.cuda.is_available()

            # A deteção é esta chamada: a rede recebe o fotograma e devolve as caixas que encontrou.
            #   conf: a confiança mínima (--conf, 0.25); abaixo dela, a deteção não se mostra;
            #   iou: quanto duas caixas da mesma classe se podem sobrepor antes de ficar só a melhor (NMS);
            #   imgsz: o tamanho a que a imagem entra na rede (640): maior é mais preciso e mais lento.
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                device=inference_device,
                imgsz=args.imgsz,
                **({"half": True} if use_half else {}),  # without CUDA, no warning per frame
                verbose=False,
            )
            t1 = time.perf_counter()

            # results tem um resultado por imagem, e aqui só há uma. Em res.boxes, cada deteção traz:
            #   xyxy: os cantos da caixa, (x1, y1) em cima à esquerda e (x2, y2) em baixo à direita, em píxeis;
            #   conf: a confiança, de 0 a 1; cls: o número da classe, que model.names passa a nome.
            # .cpu().numpy() traz os valores da placa gráfica para arrays do NumPy, que o OpenCV usa para desenhar.
            res = results[0]
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                    color = class_color(int(cls))
                    # Fancy, low-latency friendly box
                    side = max(1, min(x2 - x1, y2 - y1))
                    corner_len = max(8, min(20, side // 5))
                    draw_fancy_box(frame, x1, y1, x2, y2, color, thickness=2, length=corner_len, alpha=0.12)
                    label = f"{_class_name(names, int(cls))} {conf:.2f}"
                    draw_label(frame, x1, y1, label, color)

            # 5. Imagens por segundo: o tempo de cada volta do ciclo (ler, detetar e desenhar), com uma média
            #    suavizada para o número não saltar; «infer» é só o tempo da rede, em milissegundos.
            # FPS update (EMA across end-to-end per-frame time)
            t_now = time.perf_counter()
            dt = t_now - t_last
            t_last = t_now
            inst_fps = 1.0 / max(dt, 1e-6)
            fps = smoothing * fps + (1.0 - smoothing) * inst_fps

            if args.show_fps:
                txt = f"FPS: {fps:.1f}  (infer {1000 * (t1 - t0):.1f} ms)"
                cv2.putText(
                    frame,
                    txt,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)
            # Pace files to their nominal FPS, live sources use minimal wait
            delay_ms = 1
            if is_file_input and target_dt is not None:
                elapsed = time.perf_counter() - loop_start
                remain = max(0.0, target_dt - elapsed)
                delay_ms = max(1, int(remain * 1000.0))
            # q ou Esc sai; m espelha a imagem, para quem está à frente da câmara se ver como num espelho.
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):  # ESC or 'q'
                break
            if key == ord("m"):
                args.mirror = not args.mirror

    finally:
        try:
            if reader is not None:
                reader.release()
            else:
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

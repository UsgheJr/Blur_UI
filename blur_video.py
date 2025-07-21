import face_recognition
import cv2
import numpy as np
import torch
import os
import time
from tqdm import tqdm

# === FUNZIONE: Applica blur con maschera sfumata ===
def apply_blur_with_fade(image, x1, y1, x2, y2, kernel=(99, 99), sigma=30):
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return
    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, kernel, sigma)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.45), int(h * 0.45)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (61, 61), 0)
    mask = cv2.merge([mask] * 3).astype(float) / 255.0
    roi = roi.astype(float)
    blurred = blurred.astype(float)
    blended = (roi * (1 - mask) + blurred * mask).astype(np.uint8)
    image[y1:y2, x1:x2] = blended

# === 1. Carica modello YOLO (ottimizzato per GPU se disponibile) ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('yolov5', 'custom', path='yolov5n-license-plate.pt', source='local')
model.to(device)
model.eval()

# === 2. Input video ===
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
name = os.path.splitext(video_path)[0]
output_path = (f"Risultati/{name}_blur.mp4")
os.makedirs("Risultati", exist_ok=True)
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
YOLO_INTERVAL = 5
yolo_cache = None
start = time.time()

for _ in tqdm(range(frame_total), desc="Elaborazione video"):
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # === Sfoca volti ===
    small_rgb = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)
    face_locations = face_recognition.face_locations(small_rgb)
    for top, right, bottom, left in face_locations:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        apply_blur_with_fade(frame, left, top, right, bottom)

    # === Sfoca targhe ogni N frame ===
    if frame_count % YOLO_INTERVAL == 0:
        results = model(frame_rgb, size=640)
        yolo_cache = results.pandas().xyxy[0]

    for _, row in yolo_cache.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        apply_blur_with_fade(frame, x1, y1, x2, y2, kernel=(151, 151), sigma=50)

    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f"\nVideo sfocato salvato in: {output_path}")
print(f"Tempo totale: {time.time() - start:.2f} secondi")

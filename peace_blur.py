"""
Foto Kita Blur — trend ✌️
=========================
Real-time webcam: begitu kamu pose peace (dua jari / ✌️), seluruh frame langsung blur.

Pakai MediaPipe Tasks API (HandLandmarker). Model `hand_landmarker.task`
otomatis ke-download pas pertama kali jalan kalau belum ada.

Kontrol:
  q / ESC : keluar
  h       : show/hide HUD (teks bantu di layar)
  d       : show/hide titik landmark tangan (debug)
  [       : turunin kekuatan blur
  ]       : naikin kekuatan blur

Jalankan:
  python peace_blur.py
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Index landmark (tip, pip) tiap jari untuk cek "jari naik atau nggak".
# Referensi: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
FINGER_TIP_PIP = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def ensure_model():
    """Download model HandLandmarker kalau belum ada di folder."""
    if os.path.exists(MODEL_PATH):
        return
    print("⏬ Download model hand_landmarker.task (sekali aja)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model siap.")


def finger_is_up(lms, tip_idx, pip_idx):
    """Jari dianggap 'naik' kalau ujungnya lebih tinggi (y lebih kecil) dari sendi pip.
    Asumsi tangan menghadap ke atas — pas banget buat pose ✌️ di depan wajah."""
    return lms[tip_idx].y < lms[pip_idx].y


def is_peace(lms):
    """Peace ✌️ = telunjuk + tengah NAIK, manis + kelingking TURUN.
    Plus cek ada jarak (bentuk 'V') biar nggak ketuker sama dua jari rapat.
    `lms` = list 21 landmark dengan atribut .x, .y, .z (ternormalisasi 0..1)."""
    up = {name: finger_is_up(lms, tip, pip) for name, (tip, pip) in FINGER_TIP_PIP.items()}
    if not (up["index"] and up["middle"] and not up["ring"] and not up["pinky"]):
        return False

    # Skala tangan: jarak pergelangan (0) ke pangkal jari tengah (9).
    wrist, mid_mcp = lms[0], lms[9]
    hand_size = ((wrist.x - mid_mcp.x) ** 2 + (wrist.y - mid_mcp.y) ** 2) ** 0.5
    if hand_size == 0:
        return True  # fallback, hindari divide-by-zero

    # Jarak ujung telunjuk (8) ke ujung tengah (12) — harus cukup mekar (bentuk V).
    idx_tip, mid_tip = lms[8], lms[12]
    spread = ((idx_tip.x - mid_tip.x) ** 2 + (idx_tip.y - mid_tip.y) ** 2) ** 0.5
    return spread > 0.25 * hand_size


def draw_landmarks(frame, lms):
    """Gambar titik landmark sederhana (debug) tanpa legacy drawing_utils."""
    h, w = frame.shape[:2]
    for lm in lms:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)


def make_blur(frame, strength):
    """Blur seluruh frame. strength = ukuran kernel ganjil; makin gede makin nge-blur."""
    k = max(3, strength | 1)  # paksa ganjil
    return cv2.GaussianBlur(frame, (k, k), 0)


def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("❌ Webcam nggak kebuka. Cek izin kamera / index device.")

    # --- tuning ---
    PEACE_CONFIRM = 2   # butuh N frame berturut biar blur nyala (anti-kedip)
    HOLD_FRAMES = 6     # tahan blur beberapa frame setelah peace ilang
    blur_strength = 75  # kernel awal

    peace_streak = 0
    hold = 0
    show_hud = True
    show_landmarks = False
    last_ts = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror, biar kayak ngaca
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # timestamp ms harus selalu naik
            ts = max(last_ts + 1, int(time.perf_counter() * 1000))
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            detected = False
            if result.hand_landmarks:
                for lms in result.hand_landmarks:
                    if is_peace(lms):
                        detected = True
                    if show_landmarks:
                        draw_landmarks(frame, lms)

            # State machine on/off blur
            peace_streak = peace_streak + 1 if detected else 0
            if peace_streak >= PEACE_CONFIRM:
                blur_on, hold = True, HOLD_FRAMES
            elif hold > 0:
                blur_on, hold = True, hold - 1
            else:
                blur_on = False

            if blur_on:
                frame = make_blur(frame, blur_strength)

            # HUD digambar SETELAH blur biar tetap kebaca
            if show_hud:
                status = "BLUR ON" if blur_on else "tunjukin pose peace"
                color = (0, 0, 255) if blur_on else (0, 200, 0)
                cv2.putText(frame, status, (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"blur={blur_strength}  [q]uit [h]ud [d]ebug [ / ] strength",
                            (16, frame.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Foto Kita Blur — peace to blur", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):       # q atau ESC
                break
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("d"):
                show_landmarks = not show_landmarks
            elif key == ord("]"):
                blur_strength = min(199, blur_strength + 10)
            elif key == ord("["):
                blur_strength = max(5, blur_strength - 10)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()

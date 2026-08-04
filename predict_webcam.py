import cv2
import numpy as np
import os
import time
import tempfile
import urllib.request
from collections import deque, Counter
from PIL import Image, ImageEnhance
from tensorflow.keras.models import load_model

# config
MODEL_PATH = "age_gender_model.h5"

# window size
cap_w, cap_h = 1280, 720

LOCK_THRESHOLD = 0.70
SWITCH_THRESHOLD = 0.92
SWITCH_FRAMES_REQUIRED = 10
GENDER_BIAS_CORRECTION = 0.15

SMOOTH_FACTOR = 0.20
PERSISTENCE_FRAMES = 15
AGE_HISTORY_SIZE = 15
GENDER_HISTORY_SIZE = 20

MIN_FACE_RATIO = 0.25

THUMB_SIZE = 120

# best result ever (only updates if confidence is strictly higher)
best_ever_gender = None
best_ever_age = None
best_ever_conf = 0.0

# all perfect 100% results to display stacked
perfect_results = []
MAX_PERFECT_DISPLAY = 3

# screenshot analysis state
ss_face_img = None
ss_top3 = []

# load model
try:
    model = load_model(MODEL_PATH, compile=False)
    print("model loaded successfully")
except Exception as e:
    print(f"error loading model: {e}")
    exit()

# load face detector
temp_dir = tempfile.gettempdir()
cascade_path = os.path.join(temp_dir, 'haarcascade_frontalface_default.xml')

if not os.path.exists(cascade_path) or os.path.getsize(cascade_path) < 100000:
    print("downloading face classifier...")
    url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with open(cascade_path, 'wb') as f:
            f.write(response.read())
    print(f"download complete ({os.path.getsize(cascade_path)} bytes)")

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("error: could not load face classifier")
    exit()
print("face classifier loaded")

# preprocess face (matches training exactly)
def preprocess_face(face_crop_gray):
    pil_img = Image.fromarray(face_crop_gray)
    pil_img = pil_img.resize((128, 128), Image.LANCZOS)
    img_array = np.array(pil_img)
    img_array = img_array.reshape(1, 128, 128, 1).astype('float32') / 255.0
    return img_array

# convert numeric age to range label
def age_to_range(age):
    if age < 13:
        return "child"
    elif age < 20:
        return "teen"
    elif age < 30:
        return "young adult"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle age"
    else:
        return "senior"

# run multiple predictions on a single face crop for robust top 3
def analyze_face_deep(face_crop_gray, runs=12):
    preds = []
    for i in range(runs):
        img = Image.fromarray(face_crop_gray)
        if i == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif i == 2:
            img = ImageEnhance.Brightness(img).enhance(1.15)
        elif i == 3:
            img = ImageEnhance.Brightness(img).enhance(0.85)
        elif i == 4:
            img = ImageEnhance.Contrast(img).enhance(1.2)
        elif i == 5:
            img = ImageEnhance.Contrast(img).enhance(0.8)
        elif i == 6:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = ImageEnhance.Brightness(img).enhance(1.1)
        elif i == 7:
            img = img.resize((120, 120), Image.LANCZOS)
            img = img.resize((128, 128), Image.LANCZOS)
        elif i == 8:
            img = ImageEnhance.Sharpness(img).enhance(1.3)
        elif i == 9:
            img = ImageEnhance.Sharpness(img).enhance(0.7)
        elif i == 10:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = ImageEnhance.Contrast(img).enhance(1.1)

        arr = np.array(img.resize((128, 128), Image.LANCZOS))
        arr = arr.reshape(1, 128, 128, 1).astype('float32') / 255.0
        pred = model.predict(arr, verbose=0)
        raw_gender = float(pred[0][0][0])
        raw_age = float(pred[1][0][0])
        raw_gender = np.clip(raw_gender - GENDER_BIAS_CORRECTION, 0.0, 1.0)
        preds.append((raw_gender, raw_age))
    return preds

# draw rounded panel with shadow
def draw_panel(frame, x, y, w, h, bg_color=(30, 30, 30), alpha=0.82):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

# app state
last_face = None
frames_without_face = 0

age_history = deque(maxlen=AGE_HISTORY_SIZE)
gender_history = deque(maxlen=GENDER_HISTORY_SIZE)

locked_gender = None
locked_confidence = 0.0
switch_counter = 0

all_predictions = []
snapshot_face = None
snapshot_active = False

fps = 0
fps_counter = 0
fps_time = time.time()

# start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("error: could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)

cv2.namedWindow('age & gender prediction', cv2.WINDOW_NORMAL)
cv2.resizeWindow('age & gender prediction', cap_w, cap_h)

print("webcam started")
print("press 'q' to quit")
print("press 's' to capture and analyze screenshot")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # fps counter
    fps_counter += 1
    if time.time() - fps_time >= 1:
        fps = fps_counter
        fps_counter = 0
        fps_time = time.time()

    # face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_proc = clahe.apply(gray)

    faces = face_cascade.detectMultiScale(
        gray_proc,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(150, 150),
        maxSize=(w, h)
    )

    best_face = None
    if len(faces) > 0:
        best_face = max(faces, key=lambda f: f[2] * f[3])
        frames_without_face = 0
    else:
        frames_without_face += 1

    if best_face is None and frames_without_face < PERSISTENCE_FRAMES and last_face is not None:
        best_face = last_face

    face_crop_gray = None
    x = y = fw = fh = 0

    if best_face is not None:
        x, y, fw, fh = best_face

        # smooth rectangle
        if last_face is not None:
            lx, ly, lw, lh = last_face
            x = int(SMOOTH_FACTOR * x + (1 - SMOOTH_FACTOR) * lx)
            y = int(SMOOTH_FACTOR * y + (1 - SMOOTH_FACTOR) * ly)
            fw = int(SMOOTH_FACTOR * fw + (1 - SMOOTH_FACTOR) * lw)
            fh = int(SMOOTH_FACTOR * fh + (1 - SMOOTH_FACTOR) * lh)

        # enforce minimum face size
        min_face_size = int(min(h, w) * MIN_FACE_RATIO)
        if fw < min_face_size or fh < min_face_size:
            cx, cy = x + fw // 2, y + fh // 2
            fw = max(fw, min_face_size)
            fh = max(fh, min_face_size)
            x, y = cx - fw // 2, cy - fh // 2

        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)

        last_face = (x, y, fw, fh)

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)

        face_crop_gray = gray[y:y+fh, x:x+fw]

        try:
            face_array = preprocess_face(face_crop_gray)
            pred = model.predict(face_array, verbose=0)

            raw_gender = float(pred[0][0][0])
            raw_age = float(pred[1][0][0])

            raw_gender = np.clip(raw_gender - GENDER_BIAS_CORRECTION, 0.0, 1.0)

            gender_history.append(raw_gender)
            smooth_gender = float(np.mean(gender_history))

            age_history.append(raw_age)
            smooth_age = int(round(np.mean(age_history)))

            # gender lock system
            if smooth_gender >= 0.5:
                candidate = 'female'
                candidate_conf = smooth_gender
            else:
                candidate = 'male'
                candidate_conf = 1.0 - smooth_gender

            if locked_gender is None:
                if candidate_conf >= LOCK_THRESHOLD:
                    locked_gender = candidate
                    locked_confidence = candidate_conf
                    switch_counter = 0
            else:
                if candidate != locked_gender:
                    if candidate_conf >= SWITCH_THRESHOLD:
                        switch_counter += 1
                        if switch_counter >= SWITCH_FRAMES_REQUIRED:
                            locked_gender = candidate
                            locked_confidence = candidate_conf
                            switch_counter = 0
                    else:
                        switch_counter = 0
                else:
                    locked_confidence = 0.7 * locked_confidence + 0.3 * candidate_conf
                    switch_counter = 0

            # update best ever (only if strictly higher confidence)
            if locked_gender is not None and locked_confidence > best_ever_conf:
                best_ever_conf = locked_confidence
                best_ever_gender = locked_gender
                best_ever_age = smooth_age

            # store perfect 100% results for stacked display
            if locked_gender is not None and locked_confidence >= 0.995:
                entry = (locked_gender, smooth_age, locked_confidence)
                if not perfect_results or perfect_results[-1][:2] != entry[:2]:
                    perfect_results.append(entry)
                    if len(perfect_results) > MAX_PERFECT_DISPLAY:
                        perfect_results.pop(0)

            # main label on face (single line, centered above face)
            if best_ever_gender is not None:
                age_range = age_to_range(best_ever_age)
                main_label = f"{best_ever_gender}  |  {age_range}  |  {best_ever_age}y  |  {best_ever_conf*100:.1f}%"

                if best_ever_conf >= 0.90:
                    color = (0, 255, 0)
                elif best_ever_conf >= 0.75:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
            else:
                main_label = "detecting..."
                color = (0, 165, 255)

            (tw, th), _ = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx = x + (fw - tw) // 2
            ty = y - 14
            if ty < th + 10:
                ty = y + fh + th + 10

            draw_panel(frame, tx - 8, ty - th - 6, tw + 16, th + 12, bg_color=(10, 10, 10), alpha=0.9)
            cv2.putText(frame, main_label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # live top 3 data
            if locked_gender is not None:
                all_predictions.append((locked_gender, smooth_age))

                if not snapshot_active:
                    face_crop_color = frame[y:y+fh, x:x+fw]
                    if face_crop_color.size > 0:
                        snapshot_face = cv2.resize(face_crop_color, (THUMB_SIZE, THUMB_SIZE))
                        snapshot_active = True

        except Exception as e:
            print(f"processing error: {e}")

    else:
        if frames_without_face >= PERSISTENCE_FRAMES:
            locked_gender = None
            locked_confidence = 0.0
            switch_counter = 0
            age_history.clear()
            gender_history.clear()
            snapshot_face = None
            snapshot_active = False

    # ============================================
    # layout: organized corners
    # ============================================
    margin = 15
    line_h = 22
    small_font = 0.5
    title_font = 0.6

    # --- top-left: screenshot analysis ---
    if ss_face_img is not None:
        ss_x = margin
        ss_y = margin

        cv2.rectangle(frame, (ss_x - 2, ss_y - 2), (ss_x + THUMB_SIZE + 2, ss_y + THUMB_SIZE + 2), (200, 200, 200), 2)
        frame[ss_y:ss_y+THUMB_SIZE, ss_x:ss_x+THUMB_SIZE] = ss_face_img

        if ss_top3:
            pnl_w = 280
            pnl_h = 110
            px = ss_x
            py = ss_y + THUMB_SIZE + 8
            draw_panel(frame, px, py, pnl_w, pnl_h)

            cv2.putText(frame, "screenshot analysis", (px + 10, py + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, title_font, (200, 200, 200), 2)

            for i, (gender, age, pct) in enumerate(ss_top3[:3]):
                age_range = age_to_range(age)
                text = f"{i+1}. {gender}, {age_range} {age}y ({pct:.1f}%)"
                cv2.putText(frame, text, (px + 10, py + 48 + i * line_h),
                            cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 255, 127), 1)

    # --- top-right: live top 3 ---
    if all_predictions:
        counter = Counter(all_predictions)
        total = len(all_predictions)
        top3 = counter.most_common(3)

        pnl_w = 300
        pnl_h = 120
        px = w - pnl_w - margin
        py = margin
        draw_panel(frame, px, py, pnl_w, pnl_h)

        cv2.putText(frame, "live top 3", (px + 10, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, title_font, (200, 200, 200), 2)

        for i, ((gender, age), count) in enumerate(top3):
            pct = (count / total) * 100
            age_range = age_to_range(age)
            text = f"{i+1}. {gender}, {age_range} {age}y ({pct:.1f}%)"
            cv2.putText(frame, text, (px + 10, py + 48 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 255, 127), 1)

    # --- bottom-left: perfect results stack ---
    if perfect_results:
        pnl_w = 280
        pnl_h = 30 + len(perfect_results) * line_h
        px = margin
        py = h - pnl_h - margin - 20
        draw_panel(frame, px, py, pnl_w, pnl_h)

        cv2.putText(frame, "perfect records", (px + 10, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, title_font, (255, 215, 0), 2)

        for idx, (pg, pa, pc) in enumerate(reversed(perfect_results)):
            pr = age_to_range(pa)
            text = f"{pg}, {pr} {pa}y (100.0%)"
            cv2.putText(frame, text, (px + 10, py + 42 + idx * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 255, 0), 1)

    # --- bottom-right: face snapshot ---
    if snapshot_face is not None:
        sx = w - THUMB_SIZE - margin
        sy = h - THUMB_SIZE - margin - 20
        cv2.rectangle(frame, (sx - 2, sy - 2), (sx + THUMB_SIZE + 2, sy + THUMB_SIZE + 2), (200, 200, 200), 2)
        frame[sy:sy+THUMB_SIZE, sx:sx+THUMB_SIZE] = snapshot_face
        label_snap = "face snapshot"
        (tw, th), _ = cv2.getTextSize(label_snap, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, label_snap, (sx + (THUMB_SIZE - tw)//2, sy + THUMB_SIZE + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # --- bottom bar: controls ---
    bar_h = 28
    draw_panel(frame, 0, h - bar_h, w, bar_h, bg_color=(15, 15, 15), alpha=0.85)
    cv2.putText(frame, f"fps: {fps}   |   press 'q' to quit   |   's' to analyze screenshot",
                (15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # low light warning
    brightness = np.mean(gray)
    if brightness < 60:
        warn = "low light detected"
        (tw, th), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        wx = (w - tw) // 2
        wy = h // 2
        draw_panel(frame, wx - 10, wy - th - 5, tw + 20, th + 14, bg_color=(0, 0, 0), alpha=0.8)
        cv2.putText(frame, warn, (wx, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    display_frame = cv2.resize(frame, (cap_w, cap_h))
    cv2.imshow('age & gender prediction', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('s'):
        filename = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"screenshot saved: {filename}")

        if face_crop_gray is not None and face_crop_gray.size > 0:
            print("analyzing screenshot face...")
            deep_preds = analyze_face_deep(face_crop_gray, runs=12)

            results = []
            for rg, ra in deep_preds:
                if rg >= 0.5:
                    g = 'female'
                    conf = rg
                else:
                    g = 'male'
                    conf = 1.0 - rg
                age = int(round(ra))
                results.append((g, age, conf))

            agg = {}
            for g, a, c in results:
                k = (g, a)
                if k not in agg:
                    agg[k] = []
                agg[k].append(c)

            scored = []
            for (g, a), confs in agg.items():
                avg_conf = np.mean(confs)
                count = len(confs)
                scored.append((g, a, avg_conf, count))

            scored.sort(key=lambda x: x[2], reverse=True)

            ss_face_img = cv2.resize(frame[y:y+fh, x:x+fw], (THUMB_SIZE, THUMB_SIZE))
            ss_top3 = []
            total_runs = len(deep_preds)
            for g, a, conf, cnt in scored[:3]:
                pct = (cnt / total_runs) * 100
                ss_top3.append((g, a, pct))

            print(f"ss top 1: {ss_top3[0][0]}, {age_to_range(ss_top3[0][1])} {ss_top3[0][1]}y")
        else:
            print("no face detected to analyze")

# cleanup
cap.release()
cv2.destroyAllWindows()
print("webcam closed")
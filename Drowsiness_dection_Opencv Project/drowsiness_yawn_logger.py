import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
import platform
import csv
from datetime import datetime

# ========== CONFIGURABLE PARAMETERS ==========
# Eye Aspect Ratio
EYE_AR_THRESH = 0.23            # EAR below this → eyes considered closed
EYE_AR_CONSEC_FRAMES = 20       # frames required to trigger drowsiness alarm

# Mouth Aspect Ratio for yawning
MOUTH_AR_THRESH = 0.60          # MAR above this → mouth considered open (yawn)
MOUTH_AR_CONSEC_FRAMES = 15     # frames required to trigger yawning detection

# CSV log filename
LOG_FILENAME = "events_log.csv"

# Optional: Beep support (Windows only)
USE_BEEP = True if platform.system() == "Windows" else False
if USE_BEEP:
    import winsound

# ========== HELPER FUNCTIONS ==========

def eye_aspect_ratio(eye):
    # eye: 6 (x, y) points
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Using outer lip: 48–60 (12 points)
    # Points: 49, 53, 51, 57, 50, 58, 52, 56, 48, 54
    # A and B are vertical distances, C is horizontal
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def log_event(event_type, ear_value, mar_value):
    """
    Log event to CSV: timestamp, event, ear, mar
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILENAME, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, event_type, f"{ear_value:.3f}", f"{mar_value:.3f}"])
    print(f"[LOG] {event_type} at {now} (EAR={ear_value:.3f}, MAR={mar_value:.3f})")

# ========== LANDMARK INDICES ==========
# Eyes
RIGHT_EYE_START, RIGHT_EYE_END = 36, 42   # 36–41
LEFT_EYE_START, LEFT_EYE_END = 42, 48    # 42–47
# Mouth (outer): 48–60
MOUTH_START, MOUTH_END = 48, 61

# ========== CHECK/CREATE CSV HEADER ==========
try:
    # If file doesn't exist, this will throw
    with open(LOG_FILENAME, mode="x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event", "ear", "mar"])
except FileExistsError:
    # File already exists → do nothing
    pass

# ========== INITIALIZE DLIB MODELS ==========
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ========== START VIDEO CAPTURE ==========
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# Counters and flags
eye_counter = 0
mouth_counter = 0
drowsy_alarm_on = False
yawn_alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    # Default values if no face is found
    ear = 0.0
    mar = 0.0
    drowsy_detected = False
    yawn_detected = False

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # Eyes
        leftEye = shape[LEFT_EYE_START:LEFT_EYE_END]
        rightEye = shape[RIGHT_EYE_START:RIGHT_EYE_END]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Mouth
        mouth = shape[MOUTH_START:MOUTH_END]
        mar = mouth_aspect_ratio(mouth)

        # Draw mouth contour
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

        # --- DROWSINESS BY EYE CLOSURE ---
        if ear < EYE_AR_THRESH:
            eye_counter += 1
        else:
            eye_counter = 0
            drowsy_alarm_on = False

        if eye_counter >= EYE_AR_CONSEC_FRAMES:
            drowsy_detected = True
            if not drowsy_alarm_on:
                drowsy_alarm_on = True
                if USE_BEEP:
                    winsound.Beep(2500, 800)  # frequency, duration

        # --- YAWNING BY MOUTH OPENING ---
        if mar > MOUTH_AR_THRESH:
            mouth_counter += 1
        else:
            mouth_counter = 0
            yawn_alarm_on = False

        if mouth_counter >= MOUTH_AR_CONSEC_FRAMES:
            yawn_detected = True
            if not yawn_alarm_on:
                yawn_alarm_on = True
                if USE_BEEP:
                    winsound.Beep(1800, 600)

        # --- VISUAL TEXT OVERLAYS ---
        if drowsy_detected:
            cv2.putText(frame, "DROWSY!", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if yawn_detected:
            cv2.putText(frame, "YAWNING!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

        # Display EAR and MAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- LOG EVENTS (ONCE PER EPISODE) ---
        if drowsy_detected or yawn_detected:
            if drowsy_detected and yawn_detected:
                event_type = "BOTH"
            elif drowsy_detected:
                event_type = "DROWSY"
            else:
                event_type = "YAWN"

            # Log once when each alarm triggers
            if (drowsy_detected and drowsy_alarm_on and eye_counter == EYE_AR_CONSEC_FRAMES) or \
               (yawn_detected and yawn_alarm_on and mouth_counter == MOUTH_AR_CONSEC_FRAMES) or \
               (event_type == "BOTH" and (
                    eye_counter == EYE_AR_CONSEC_FRAMES or mouth_counter == MOUTH_AR_CONSEC_FRAMES
                )):
                log_event(event_type, ear, mar)

    cv2.imshow("Driver Drowsiness & Yawn Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

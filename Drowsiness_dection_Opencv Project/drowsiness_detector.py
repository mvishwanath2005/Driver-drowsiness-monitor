import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
import platform
import requests   # <<< added for ESP8266 communication

# ================== CONFIG ==================
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 20

ESP_IP = "192.168.4.1"   # <<< change if your ESP IP is different
SEND_STOP_ON_ALERT = True

USE_BEEP = True if platform.system() == "Windows" else False
if USE_BEEP:
    import winsound
# ============================================


# ========== ROBOT CONTROL ==========
def stop_robot():
    if not SEND_STOP_ON_ALERT:
        return
    try:
        requests.get(f"http://{ESP_IP}/STOP", timeout=1)
        print("[INFO] STOP sent to robot")
    except Exception as e:
        print("[WARN] Could not contact robot:", e)


# ========== HELPER FUNCTIONS ==========
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


RIGHT_EYE_START, RIGHT_EYE_END = 36, 42
LEFT_EYE_START, LEFT_EYE_END = 42, 48


# ========== LOAD MODELS ==========
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)


COUNTER = 0
ALARM_ON = False


# ========== MAIN LOOP ==========
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera frame missing")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        leftEye = shape[LEFT_EYE_START:LEFT_EYE_END]
        rightEye = shape[RIGHT_EYE_START:RIGHT_EYE_END]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        # ====== Drowsiness Logic ======
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    print("[ALERT] Drowsiness detected!")

                    stop_robot()   # <<< ROBOT STOP HERE

                    if USE_BEEP:
                        winsound.Beep(2500, 1000)

                cv2.putText(frame, "DROWSY!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255), 3)

        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

import os
import sys
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
import csv
from datetime import datetime
import platform
import threading

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk  # pip install pillow


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller .exe
    """
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ========== CONFIG ==========

# Where to store logs & exported reports (user Documents -> DrowsinessMonitor)
LOG_DIR = os.path.join(os.path.expanduser("~"), "Documents", "DrowsinessMonitor")
LOG_FILENAME = os.path.join(LOG_DIR, "events_log.csv")

MODEL_PATH = resource_path("shape_predictor_68_face_landmarks.dat")

# How long (in frames) we allow NO FACE before auto-pausing
NO_FACE_CONSEC_FRAMES = 60  # ~2 sec at 30 FPS

USE_BEEP = True if platform.system() == "Windows" else False
if USE_BEEP:
    import winsound


# ========== HELPER FUNCTIONS ==========

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    # Outer mouth landmarks (48–60)
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


def ensure_csv_header():
    # Make sure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        with open(LOG_FILENAME, mode="x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "ear", "mar"])
    except FileExistsError:
        pass


def log_event(event_type, ear_value, mar_value):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILENAME, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, event_type, f"{ear_value:.3f}", f"{mar_value:.3f}"])


def format_duration(td):
    """Return HH:MM:SS for a timedelta."""
    total_sec = int(td.total_seconds())
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ========== SOUND MANAGER (REAL-TIME ALERTS) ==========

class SoundManager:
    """
    Handles sound alerts.
    - Continuous alarm for drowsiness (real-time)
    - Short beep for yawns
    """
    def __init__(self, is_enabled_callback, is_drowsy_callback):
        self.is_enabled_callback = is_enabled_callback
        self.is_drowsy_callback = is_drowsy_callback

        self.yawn_playing = False
        self.lock = threading.Lock()

        # Start continuous drowsy alarm loop in background
        t = threading.Thread(target=self._drowsy_loop, daemon=True)
        t.start()

    def _can_play(self):
        return USE_BEEP and self.is_enabled_callback()

    def _drowsy_loop(self):
        """
        Continuous real-time loop:
        If drowsy AND sound enabled -> keep beeping.
        Stops automatically when drowsy flag becomes False.
        """
        while True:
            try:
                if self._can_play() and self.is_drowsy_callback():
                    winsound.Beep(1600, 300)  # 300 ms beep
                else:
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    def _play_pattern(self, pattern, flag_attr):
        """
        pattern: list of (freq, duration_ms, pause_ms)
        flag_attr: 'yawn_playing'
        """
        try:
            for freq, dur, pause in pattern:
                if not self._can_play():
                    break
                winsound.Beep(freq, dur)
                if pause > 0:
                    time.sleep(pause / 1000.0)
        except Exception:
            pass
        finally:
            with self.lock:
                setattr(self, flag_attr, False)

    def trigger_yawn_alarm(self):
        """
        Short tone for yawning (once per yawn event).
        """
        if not self._can_play():
            return

        with self.lock:
            if self.yawn_playing:
                return
            self.yawn_playing = True

        pattern = [
            (1200, 200, 40),
            (900, 200, 0),
        ]

        t = threading.Thread(target=self._play_pattern, args=(pattern, "yawn_playing"), daemon=True)
        t.start()

    def stop_all(self):
        with self.lock:
            self.yawn_playing = False
        # drowsy loop auto-silences when is_drowsy_callback() becomes False


# ========== MAIN APP CLASS ==========

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness & Safety Monitoring")
        self.root.geometry("1000x620")
        self.root.resizable(False, False)

        # --- State variables ---
        self.cap = None
        self.running = False
        self.processing_thread = None

        # Auto-pause (no driver)
        self.no_face_counter = 0
        self.no_driver_paused = False

        # Session timing
        self.session_start_time = None
        self.session_end_time = None
        self.first_event_time = None
        self.last_event_time = None

        # Shared data
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self.ear = 0.0
        self.mar = 0.0

        self.eye_counter = 0
        self.mouth_counter = 0

        self.drowsy_events = 0
        self.yawn_events = 0

        self.drowsy_alarm_on = False   # used by real-time sound
        self.yawn_alarm_on = False

        self.current_status_text = "IDLE"
        self.current_status_color = "gray"

        # Sensitivity profiles (yawn more sensitive)
        self.sensitivity_profiles = {
            "High":   {"ear_thresh": 0.25, "ear_frames": 15, "mar_thresh": 0.40, "mar_frames": 6},
            "Normal": {"ear_thresh": 0.23, "ear_frames": 20, "mar_thresh": 0.45, "mar_frames": 8},
            "Low":    {"ear_thresh": 0.21, "ear_frames": 25, "mar_thresh": 0.50, "mar_frames": 10},
        }
        self.current_profile_name = tk.StringVar(value="Normal")

        self.eye_ar_thresh = self.sensitivity_profiles["Normal"]["ear_thresh"]
        self.eye_ar_frames = self.sensitivity_profiles["Normal"]["ear_frames"]
        self.mouth_ar_thresh = self.sensitivity_profiles["Normal"]["mar_thresh"]
        self.mouth_ar_frames = self.sensitivity_profiles["Normal"]["mar_frames"]

        self.beep_enabled = tk.BooleanVar(value=True)

        # Vehicle dashboard
        self.speed = tk.IntVar(value=60)   # km/h
        self.safety_score = 100            # 0–100

        # Dlib models
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading landmark model:\n{e}")
            raise

        # Landmark indices
        self.RIGHT_EYE_START, self.RIGHT_EYE_END = 36, 42
        self.LEFT_EYE_START, self.LEFT_EYE_END = 42, 48
        self.MOUTH_START, self.MOUTH_END = 48, 61

        ensure_csv_header()

        # Sound manager (real-time alerts)
        self.sound_manager = SoundManager(self._is_beep_enabled, self._is_drowsy_active)

        self._build_ui()

    def _is_beep_enabled(self):
        return self.beep_enabled.get()

    def _is_drowsy_active(self):
        return self.drowsy_alarm_on

    # ========== UI BUILD ==========

    def _build_ui(self):
        # Video
        video_frame = ttk.LabelFrame(self.root, text="Live View")
        video_frame.place(x=10, y=10, width=640, height=480)

        self.video_label = ttk.Label(video_frame)
        self.video_label.place(x=0, y=0, width=640, height=450)

        # Right panel
        ctrl_frame = ttk.LabelFrame(self.root, text="Controls & Status")
        ctrl_frame.place(x=660, y=10, width=320, height=580)

        # Start / Stop
        self.start_btn = ttk.Button(ctrl_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_btn.pack(pady=(10, 5), fill="x", padx=10)

        self.stop_btn = ttk.Button(
            ctrl_frame,
            text="Stop Monitoring",
            command=self.stop_monitoring,
            state="disabled"
        )
        self.stop_btn.pack(pady=5, fill="x", padx=10)

        # Sensitivity
        ttk.Label(ctrl_frame, text="Sensitivity").pack(pady=(10, 5), anchor="w", padx=10)
        self.sens_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.current_profile_name,
            values=list(self.sensitivity_profiles.keys()),
            state="readonly"
        )
        self.sens_combo.pack(pady=5, fill="x", padx=10)
        self.sens_combo.bind("<<ComboboxSelected>>", self.on_sensitivity_change)

        # Beep checkbox
        self.beep_check = ttk.Checkbutton(
            ctrl_frame,
            text="Enable Sound Alerts",
            variable=self.beep_enabled
        )
        self.beep_check.pack(pady=5, anchor="w", padx=10)

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", pady=8)

        # Status
        ttk.Label(ctrl_frame, text="System Status:").pack(anchor="w", padx=10)
        self.status_label = ttk.Label(ctrl_frame, text="IDLE", foreground="gray")
        self.status_label.pack(anchor="w", padx=20, pady=(0, 8))

        ttk.Label(ctrl_frame, text="EAR (Eye Aspect Ratio):").pack(anchor="w", padx=10)
        self.ear_label = ttk.Label(ctrl_frame, text="0.00")
        self.ear_label.pack(anchor="w", padx=20)

        ttk.Label(ctrl_frame, text="MAR (Mouth Aspect Ratio):").pack(anchor="w", padx=10, pady=(8, 0))
        self.mar_label = ttk.Label(ctrl_frame, text="0.00")
        self.mar_label.pack(anchor="w", padx=20)

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", pady=8)

        # Events
        ttk.Label(ctrl_frame, text="Drowsy Events:").pack(anchor="w", padx=10)
        self.drowsy_count_label = ttk.Label(ctrl_frame, text="0")
        self.drowsy_count_label.pack(anchor="w", padx=20)

        ttk.Label(ctrl_frame, text="Yawn Events:").pack(anchor="w", padx=10, pady=(8, 0))
        self.yawn_count_label = ttk.Label(ctrl_frame, text="0")
        self.yawn_count_label.pack(anchor="w", padx=20)

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", pady=8)

        # Speed + Safety
        ttk.Label(ctrl_frame, text="Vehicle Speed (km/h):").pack(anchor="w", padx=10)
        self.speed_value_label = ttk.Label(ctrl_frame, text=f"{self.speed.get()} km/h")
        self.speed_value_label.pack(anchor="w", padx=20)

        self.speed_scale = ttk.Scale(
            ctrl_frame,
            from_=0,
            to=140,
            orient="horizontal",
            variable=self.speed,
            command=self.on_speed_change
        )
        self.speed_scale.pack(fill="x", padx=10, pady=(5, 5))

        ttk.Label(ctrl_frame, text="Safety Score:").pack(anchor="w", padx=10, pady=(8, 0))
        self.safety_score_label = ttk.Label(ctrl_frame, text="100 / 100")
        self.safety_score_label.pack(anchor="w", padx=20)

        self.safety_canvas = tk.Canvas(
            ctrl_frame,
            width=270,
            height=22,
            bg="#222222",
            highlightthickness=0
        )
        self.safety_canvas.pack(padx=10, pady=(5, 5))

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", pady=8)

        # Data export
        ttk.Label(ctrl_frame, text="Data Export (for Cloud / Report):").pack(anchor="w", padx=10)

        export_frame1 = ttk.Frame(ctrl_frame)
        export_frame1.pack(fill="x", padx=10, pady=(5, 2))

        self.export_csv_btn = ttk.Button(
            export_frame1,
            text="Export CSV",
            command=self.export_session_csv
        )
        self.export_csv_btn.pack(side="left", expand=True, fill="x")

        self.export_json_btn = ttk.Button(
            export_frame1,
            text="Export JSON",
            command=self.export_session_json
        )
        self.export_json_btn.pack(side="left", expand=True, fill="x", padx=(5, 0))

        export_frame2 = ttk.Frame(ctrl_frame)
        export_frame2.pack(fill="x", padx=10, pady=(2, 5))

        self.export_html_btn = ttk.Button(
            export_frame2,
            text="Export HTML Report",
            command=self.export_session_html
        )
        self.export_html_btn.pack(side="left", expand=True, fill="x")

        # Footer
        footer = ttk.Label(
            self.root,
            text="Start monitoring → real-time alerts → export CSV/JSON/HTML (saved in Documents\\DrowsinessMonitor).",
            anchor="center"
        )
        footer.place(x=10, y=580, width=980, height=30)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ========== UI ACTIONS ==========

    def on_sensitivity_change(self, event=None):
        profile_name = self.current_profile_name.get()
        profile = self.sensitivity_profiles.get(
            profile_name,
            self.sensitivity_profiles["Normal"]
        )
        self.eye_ar_thresh = profile["ear_thresh"]
        self.eye_ar_frames = profile["ear_frames"]
        self.mouth_ar_thresh = profile["mar_thresh"]
        self.mouth_ar_frames = profile["mar_frames"]

    def on_speed_change(self, value):
        self.speed_value_label.config(text=f"{self.speed.get()} km/h")

    def start_monitoring(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        # reset counters and times
        self.eye_counter = 0
        self.mouth_counter = 0
        self.drowsy_alarm_on = False
        self.yawn_alarm_on = False
        self.drowsy_events = 0
        self.yawn_events = 0
        self.first_event_time = None
        our_last = None
        self.last_event_time = None

        self.session_start_time = datetime.now()
        self.session_end_time = None

        self.no_face_counter = 0
        self.no_driver_paused = False

        self.current_status_text = "RUNNING"
        self.current_status_color = "green"

        # Start background processing thread
        self.processing_thread = threading.Thread(
            target=self.process_loop,
            daemon=True
        )
        self.processing_thread.start()

        # Start GUI update loop
        self.update_gui()

    def stop_monitoring(self):
        if not self.running:
            return

        self.session_end_time = datetime.now()

        self.running = False

        if self.processing_thread is not None:
            self.processing_thread = None

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None

        with self.frame_lock:
            self.latest_frame = None

        self.drowsy_alarm_on = False
        self.yawn_alarm_on = False
        self.sound_manager.stop_all()

        self.status_label.config(text="STOPPED", foreground="orange")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.video_label.config(image="")

        self._show_session_summary()

    def _show_session_summary(self):
        """
        Show session summary in a custom popup window
        with Export CSV / JSON / HTML buttons.
        """
        if self.session_start_time is None or self.session_end_time is None:
            return

        # build summary text
        duration = self.session_end_time - self.session_start_time
        duration_str = format_duration(duration)

        total_events = self.drowsy_events + self.yawn_events
        hours = duration.total_seconds() / 3600.0 if duration.total_seconds() > 0 else 0
        if hours > 0:
            rate = total_events / hours
            rate_str = f"{rate:.2f} events/hour"
        else:
            rate_str = "N/A"

        if self.first_event_time is not None:
            first_str = self.first_event_time.strftime("%Y-%m-%d %H:%M:%S")
            last_str = self.last_event_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            first_str = "No events"
            last_str = "No events"

        summary_text = (
            "Session Summary\n\n"
            f"Monitoring duration : {duration_str}\n"
            f"Total events        : {total_events}\n"
            f"  - Drowsy events   : {self.drowsy_events}\n"
            f"  - Yawn events     : {self.yawn_events}\n"
            f"Event rate          : {rate_str}\n\n"
            f"First event time    : {first_str}\n"
            f"Last event time     : {last_str}\n\n"
            f"Logs & reports saved in:\n{LOG_DIR}"
        )

        # custom popup window
        win = tk.Toplevel(self.root)
        win.title("Session Summary")
        win.resizable(False, False)
        win.grab_set()

        lbl = tk.Label(
            win,
            text=summary_text,
            justify="left",
            font=("Segoe UI", 9)
        )
        lbl.pack(padx=15, pady=(15, 10))

        # Export buttons
        btn_frame = ttk.Frame(win)
        btn_frame.pack(padx=15, pady=(0, 15), fill="x")

        csv_btn = ttk.Button(
            btn_frame,
            text="Export CSV",
            command=self.export_session_csv
        )
        csv_btn.pack(side="left", expand=True, fill="x")

        json_btn = ttk.Button(
            btn_frame,
            text="Export JSON",
            command=self.export_session_json
        )
        json_btn.pack(side="left", expand=True, fill="x", padx=5)

        html_btn = ttk.Button(
            btn_frame,
            text="Export HTML Report",
            command=self.export_session_html
        )
        html_btn.pack(side="left", expand=True, fill="x")

        # Close button row
        close_frame = ttk.Frame(win)
        close_frame.pack(padx=15, pady=(0, 10), fill="x")

        close_btn = ttk.Button(close_frame, text="Close", command=win.destroy)
        close_btn.pack(side="right")

    def on_close(self):
        if self.running:
            self.stop_monitoring()
        self.root.destroy()

    # ========== CORE PROCESSING (BACKGROUND THREAD) ==========

    def _record_event_time(self):
        now = datetime.now()
        if self.first_event_time is None:
            self.first_event_time = now
        self.last_event_time = now

    def process_loop(self):
        while self.running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = self.detector(gray, 0)

            ear = 0.0
            mar = 0.0
            drowsy_detected = False
            yawn_detected = False

            if len(rects) > 0:
                # Reset "no face" counter and unpause if needed
                self.no_face_counter = 0
                if self.no_driver_paused:
                    self.no_driver_paused = False

                rect = rects[0]
                shape = self.predictor(gray, rect)
                shape = shape_to_np(shape)

                # Eyes
                leftEye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
                rightEye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # Mouth
                mouth = shape[self.MOUTH_START:self.MOUTH_END]
                mar = mouth_aspect_ratio(mouth)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

                # Only process drowsiness/yawn if NOT paused
                if not self.no_driver_paused:
                    # Drowsiness (eyes)
                    if ear < self.eye_ar_thresh:
                        self.eye_counter += 1
                    else:
                        self.eye_counter = 0
                        self.drowsy_alarm_on = False  # turn off alarm when eyes open

                    if self.eye_counter >= self.eye_ar_frames:
                        drowsy_detected = True
                        self.drowsy_alarm_on = True
                        if self.eye_counter == self.eye_ar_frames:
                            self.drowsy_events += 1
                            self._record_event_time()
                            log_event("DROWSY", ear, mar)

                    # Yawning (mouth)
                    if mar > self.mouth_ar_thresh:
                        self.mouth_counter += 1
                    else:
                        self.mouth_counter = 0
                        self.yawn_alarm_on = False

                    if self.mouth_counter >= self.mouth_ar_frames:
                        yawn_detected = True
                        if not self.yawn_alarm_on:
                            self.yawn_alarm_on = True
                            self.yawn_events += 1
                            self._record_event_time()
                            log_event("YAWN", ear, mar)
                            self.sound_manager.trigger_yawn_alarm()

                # Text overlays
                if drowsy_detected:
                    cv2.putText(
                        frame,
                        "DROWSY!",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3
                    )
                if yawn_detected:
                    cv2.putText(
                        frame,
                        "YAWNING!",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 0, 255),
                        3
                    )

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                # Status
                if self.no_driver_paused:
                    self.current_status_text = "PAUSED (NO DRIVER)"
                    self.current_status_color = "orange"
                else:
                    if drowsy_detected:
                        self.current_status_text = "DROWSY"
                        self.current_status_color = "red"
                    elif yawn_detected:
                        self.current_status_text = "YAWNING"
                        self.current_status_color = "purple"
                    else:
                        self.current_status_text = "RUNNING"
                        self.current_status_color = "green"
            else:
                # No face detected
                self.no_face_counter += 1

                self.drowsy_alarm_on = False  # no driver -> no drowsy alarm sound

                if self.no_face_counter >= NO_FACE_CONSEC_FRAMES:
                    self.no_driver_paused = True
                    self.current_status_text = "PAUSED (NO DRIVER)"
                    self.current_status_color = "orange"
                else:
                    self.current_status_text = "NO DRIVER"
                    self.current_status_color = "red"

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Update shared metrics
            self.ear = ear
            self.mar = mar

            with self.frame_lock:
                self.latest_frame = frame.copy()

            time.sleep(0.005)

        self.running = False
        self.drowsy_alarm_on = False
        self.yawn_alarm_on = False

    # ========== SAFETY SCORE & GAUGE ==========

    def _calculate_safety_score(self):
        if self.no_driver_paused:
            self.safety_score = 100
            return

        score = 100

        if self.current_status_text == "DROWSY":
            score -= 40
        elif self.current_status_text == "YAWNING":
            score -= 20

        total_events = self.drowsy_events + self.yawn_events
        score -= min(total_events * 3, 30)

        v = self.speed.get()
        if v > 100:
            score -= 25
        elif v > 80:
            score -= 15
        elif v > 60:
            score -= 8

        score = max(0, min(100, score))
        self.safety_score = score

    def _update_safety_display(self):
        self._calculate_safety_score()
        s = self.safety_score

        self.safety_score_label.config(text=f"{int(s)} / 100")

        self.safety_canvas.delete("all")
        width = 270
        height = 22
        margin = 2
        fill_width = int((width - 2 * margin) * (s / 100.0))

        if s >= 70:
            color = "#00c853"  # green
        elif s >= 40:
            color = "#ffab00"  # yellow
        else:
            color = "#d50000"  # red

        self.safety_canvas.create_rectangle(
            margin, margin, width - margin, height - margin,
            outline="#444444", fill="#111111"
        )

        if fill_width > 0:
            self.safety_canvas.create_rectangle(
                margin, margin,
                margin + fill_width, height - margin,
                outline="", fill=color
            )

        self.safety_canvas.create_rectangle(
            margin, margin, width - margin, height - margin,
            outline="#ffffff"
        )

    # ========== DATA EXPORT HELPERS ==========

    def _get_session_events(self):
        if self.session_start_time is None or self.session_end_time is None:
            return []

        rows = []
        try:
            with open(LOG_FILENAME, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        continue
                    if self.session_start_time <= ts <= self.session_end_time:
                        rows.append({
                            "timestamp": row["timestamp"],
                            "event": row["event"],
                            "ear": row["ear"],
                            "mar": row["mar"],
                        })
        except FileNotFoundError:
            pass

        return rows

    def export_session_csv(self):
        if self.session_start_time is None or self.session_end_time is None:
            messagebox.showwarning(
                "Export CSV",
                "No completed session. Start and stop monitoring first."
            )
            return

        os.makedirs(LOG_DIR, exist_ok=True)

        session_id = self.session_start_time.strftime("session_%Y%m%d_%H%M%S")
        filename = os.path.join(LOG_DIR, f"{session_id}.csv")

        rows = self._get_session_events()
        if not rows:
            messagebox.showinfo(
                "Export CSV",
                "No events recorded in this session. CSV not created."
            )
            return

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "ear", "mar"])
            for r in rows:
                writer.writerow([r["timestamp"], r["event"], r["ear"], r["mar"]])

        messagebox.showinfo(
            "Export CSV",
            f"Session data exported to:\n{filename}"
        )

    def export_session_json(self):
        import json

        if self.session_start_time is None or self.session_end_time is None:
            messagebox.showwarning(
                "Export JSON",
                "No completed session. Start and stop monitoring first."
            )
            return

        os.makedirs(LOG_DIR, exist_ok=True)

        session_id = self.session_start_time.strftime("session_%Y%m%d_%H%M%S")
        filename = os.path.join(LOG_DIR, f"{session_id}.json")

        rows = self._get_session_events()

        summary = {
            "session_start": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.session_start_time else None,
            "session_end": self.session_end_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.session_end_time else None,
            "drowsy_events": self.drowsy_events,
            "yawn_events": self.yawn_events,
            "total_events": self.drowsy_events + self.yawn_events,
        }

        data = {
            "summary": summary,
            "events": rows,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo(
            "Export JSON",
            f"Session JSON exported to:\n{filename}"
        )

    def export_session_html(self):
        """
        Generate a simple HTML report for the last completed session.
        Includes summary + event table.
        """
        if self.session_start_time is None or self.session_end_time is None:
            messagebox.showwarning(
                "Export HTML",
                "No completed session. Start and stop monitoring first."
            )
            return

        os.makedirs(LOG_DIR, exist_ok=True)

        session_id = self.session_start_time.strftime("session_%Y%m%d_%H%M%S")
        filename = os.path.join(LOG_DIR, f"{session_id}.html")

        rows = self._get_session_events()

        duration_str = "N/A"
        if self.session_end_time and self.session_start_time:
            duration = self.session_end_time - self.session_start_time
            duration_str = format_duration(duration)
        total_events = self.drowsy_events + self.yawn_events

        first_str = (
            self.first_event_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.first_event_time else "No events"
        )
        last_str = (
            self.last_event_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.last_event_time else "No events"
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Driver Drowsiness Session Report - {session_id}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    background-color: #111827;
    color: #e5e7eb;
    margin: 0;
    padding: 20px;
}}
h1 {{
    color: #38bdf8;
}}
h2 {{
    color: #a5b4fc;
}}
.section {{
    margin-bottom: 24px;
    padding: 16px;
    background-color: #1f2937;
    border-radius: 8px;
    border: 1px solid #374151;
}}
.summary-table td {{
    padding: 4px 12px;
}}
table.events {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
}}
table.events th, table.events td {{
    border: 1px solid #4b5563;
    padding: 6px 8px;
    font-size: 14px;
}}
table.events th {{
    background-color: #111827;
}}
tr.event-drowsy td {{
    background-color: #7f1d1d;
}}
tr.event-yawn td {{
    background-color: #312e81;
}}
.footer {{
    margin-top: 24px;
    font-size: 12px;
    color: #9ca3af;
    text-align: center;
}}
</style>
</head>
<body>
<h1>Driver Drowsiness & Safety Monitoring - Session Report</h1>

<div class="section">
  <h2>Session Summary</h2>
  <table class="summary-table">
    <tr><td><b>Session ID</b></td><td>{session_id}</td></tr>
    <tr><td><b>Start Time</b></td><td>{self.session_start_time.strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
    <tr><td><b>End Time</b></td><td>{self.session_end_time.strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
    <tr><td><b>Duration</b></td><td>{duration_str}</td></tr>
    <tr><td><b>Total Events</b></td><td>{total_events}</td></tr>
    <tr><td>&bull; Drowsy Events</td><td>{self.drowsy_events}</td></tr>
    <tr><td>&bull; Yawn Events</td><td>{self.yawn_events}</td></tr>
    <tr><td><b>First Event</b></td><td>{first_str}</td></tr>
    <tr><td><b>Last Event</b></td><td>{last_str}</td></tr>
  </table>
</div>

<div class="section">
  <h2>Event Timeline</h2>
"""

        if rows:
            html += """
  <table class="events">
    <thead>
      <tr>
        <th>#</th>
        <th>Timestamp</th>
        <th>Event</th>
        <th>EAR</th>
        <th>MAR</th>
      </tr>
    </thead>
    <tbody>
"""
            for idx, r in enumerate(rows, start=1):
                ev = r["event"].upper()
                row_class = ""
                if ev == "DROWSY":
                    row_class = "event-drowsy"
                elif ev == "YAWN":
                    row_class = "event-yawn"
                html += f"""      <tr class="{row_class}">
        <td>{idx}</td>
        <td>{r["timestamp"]}</td>
        <td>{ev}</td>
        <td>{r["ear"]}</td>
        <td>{r["mar"]}</td>
      </tr>
"""
            html += """    </tbody>
  </table>
"""
        else:
            html += "<p>No events recorded during this session.</p>\n"

        html += f"""
</div>

<div class="footer">
  Generated by Driver Drowsiness & Safety Monitoring System.<br>
  Logs & reports saved in: {LOG_DIR}
</div>

</body>
</html>
"""

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
            messagebox.showinfo(
                "Export HTML",
                f"HTML report exported to:\n{filename}"
            )
        except Exception as e:
            messagebox.showerror("Export HTML", f"Failed to write HTML file:\n{e}")

    # ========== GUI UPDATE LOOP ==========

    def update_gui(self):
        if not self.running:
            return

        frame_to_show = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame_to_show = self.latest_frame.copy()

        if frame_to_show is not None:
            frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.ear_label.config(text=f"{self.ear:.2f}")
        self.mar_label.config(text=f"{self.mar:.2f}")
        self.status_label.config(text=self.current_status_text, foreground=self.current_status_color)
        self.drowsy_count_label.config(text=str(self.drowsy_events))
        self.yawn_count_label.config(text=str(self.yawn_events))
        self.speed_value_label.config(text=f"{self.speed.get()} km/h")

        self._update_safety_display()

        self.root.after(30, self.update_gui)


# ========== RUN APP ==========

if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = DrowsinessApp(root)
    root.mainloop()

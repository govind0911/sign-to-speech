"""
engine.py — TSWAG Glasses
--------------------------
Owns the webcam, MediaPipe Tasks models (HandLandmarker + FaceLandmarker),
and the per-frame detection loop. Runs on a background QThread so the GUI
never freezes. Emits one signal per processed frame with everything the
window needs to draw itself; all *temporal* logic (hold-to-lock timers,
sentence building, TTS) lives in main.py, not here — this file only
answers "what do I see in THIS frame".

Model files are downloaded once (~15 MB total) from Google's public
MediaPipe model bucket the first time the app runs, then cached locally.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    RunningMode,
)

import gestures

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# Connections for drawing the hand skeleton (MediaPipe's 21-point layout)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]

# A light subset of face-mesh connections (oval + eyes + lips) so the
# overlay reads as "face tracking" without redrawing all ~468 points.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109, 10]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
LEFT_EYE = [33, 160, 158, 133, 153, 144, 33]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362]


def ensure_models(progress_cb=None):
    """Download the two .task model bundles if not already present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for path, url in ((HAND_MODEL_PATH, HAND_MODEL_URL), (FACE_MODEL_PATH, FACE_MODEL_URL)):
        if not os.path.exists(path):
            if progress_cb:
                progress_cb(f"Downloading {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path)


def list_available_cameras(max_index=5):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if cap is not None and cap.isOpened():
            available.append(i)
        cap.release()
    return available or [0]


class CameraEngine(QThread):
    frame_ready = pyqtSignal(dict)
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    recording_progress = pyqtSignal(int)   # frames remaining
    recording_done = pyqtSignal(str)       # gesture name

    def __init__(self, custom_manager, parent=None):
        super().__init__(parent)
        self.custom_manager = custom_manager
        self._running = False
        self.camera_index = 0
        self.recognition_mode = "builtin"     # "builtin" or "custom"
        self.emotion_enabled = True
        self.confidence_threshold = 0.75
        self._hand_landmarker = None
        self._face_landmarker = None
        self._record_target = None            # gesture name currently being recorded
        self._record_frames_left = 0

    def start_recording_gesture(self, name, num_frames):
        """Ask the engine to capture `num_frames` feature vectors for `name`
        on upcoming frames (used by the gesture trainer dialog)."""
        self._record_target = name.strip().upper()
        self._record_frames_left = num_frames

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        try:
            self.status.emit("Loading recognition models ...")
            ensure_models(progress_cb=self.status.emit)

            hand_opts = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            face_opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=True,
            )
            self._hand_landmarker = HandLandmarker.create_from_options(hand_opts)
            self._face_landmarker = FaceLandmarker.create_from_options(face_opts)
        except Exception as exc:
            self.error.emit(f"Could not load recognition models: {exc}")
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cap.isOpened():
            self.error.emit("Could not open the webcam. Check the camera "
                             "selection in Settings or that no other app is using it.")
            return

        self.status.emit("Camera ready")
        self._running = True
        prev_time = time.time()
        fps = 0.0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.error.emit("Webcam feed was interrupted.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            try:
                hand_result = self._hand_landmarker.detect(mp_image)
                face_result = self._face_landmarker.detect(mp_image)
            except Exception as exc:
                self.error.emit(f"Recognition error: {exc}")
                continue

            hands = []
            for i, lm_list in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[i][0].category_name
                hands.append({
                    "landmarks": lm_list,
                    "handedness": handedness,
                    "fingers": gestures.fingers_extended(lm_list, handedness),
                })

            face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None
            blendshapes = (face_result.face_blendshapes[0]
                           if face_result.face_blendshapes else None)

            self._draw_overlays(frame, hands, face_landmarks)

            word, confidence, both_open = self._recognize(hands, face_landmarks)

            # Handle an in-progress custom-gesture recording session
            if self._record_target and hands:
                if len(hands) == 1:
                    fv = gestures.hand_feature_vector(hands[0]["landmarks"])
                else:
                    fv = gestures.two_hand_feature_vector(
                        [h["landmarks"] for h in hands],
                        [h["handedness"] for h in hands],
                    )
                self.custom_manager.add_sample(self._record_target, fv)
                self._record_frames_left -= 1
                self.recording_progress.emit(self._record_frames_left)
                if self._record_frames_left <= 0:
                    self.custom_manager.finish_recording()
                    self.recording_done.emit(self._record_target)
                    self._record_target = None

            emotions = []
            if self.emotion_enabled and blendshapes is not None:
                emotions = gestures.top2_emotions(blendshapes)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).rgbSwapped().copy()

            self.frame_ready.emit({
                "image": qimg,
                "word": word,
                "confidence": confidence,
                "both_hands_open": both_open,
                "emotions": emotions,
                "fps": fps,
                "num_hands": len(hands),
            })

        cap.release()
        if self._hand_landmarker:
            self._hand_landmarker.close()
        if self._face_landmarker:
            self._face_landmarker.close()

    def _recognize(self, hands, face_landmarks):
        if not hands:
            return None, 0.0, False

        if self.recognition_mode == "custom":
            if len(hands) == 1:
                fv = gestures.hand_feature_vector(hands[0]["landmarks"])
            else:
                fv = gestures.two_hand_feature_vector(
                    [h["landmarks"] for h in hands],
                    [h["handedness"] for h in hands],
                )
            word, conf = self.custom_manager.predict(fv)
            both_open = len(hands) == 2 and all(
                gestures.is_open_palm(h["fingers"]) for h in hands
            )
            if word is not None and conf < self.confidence_threshold:
                word = None
            return word, conf, both_open

        word, conf, both_open = gestures.classify_builtin(hands, face_landmarks)
        if word is not None and conf < self.confidence_threshold:
            word = None
        return word, conf, both_open

    def _draw_overlays(self, frame, hands, face_landmarks):
        h, w = frame.shape[:2]

        for hand in hands:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand["landmarks"]]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (66, 133, 244), 2, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 3, (66, 133, 244), 1, cv2.LINE_AA)

        if face_landmarks is not None:
            for group in (FACE_OVAL, LIPS, LEFT_EYE, RIGHT_EYE):
                pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in group]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(frame, a, b, (52, 168, 235), 1, cv2.LINE_AA)

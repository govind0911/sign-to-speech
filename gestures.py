"""
gestures.py — TSWAG Glasses
----------------------------
Everything related to turning raw hand/face landmarks into a recognized
"word". Contains:

  * Landmark -> feature-vector math (used by both the built-in and the
    custom/trainable recognizers so they share one geometry pipeline).
  * BUILTIN mode: hand-crafted rules for the six required words
    (USE, TECHNOLOGY, FOR, HELP, NOT, WAR) plus a "both hands open" pose
    used to trigger sentence read-back. These are simplified,
    easy-to-perform poses inspired by real ASL handshapes (U, T, W, A,
    fist+flat-palm, pointing) rather than a full ASL classifier — real
    continuous sign-language recognition needs a trained video model,
    which is out of scope for a lightweight offline desktop app. The
    Custom mode below is what lets a user teach the app *real* signs
    (or anything else) if the built-in set isn't accurate enough for them.
  * CUSTOM mode: nearest-centroid classifier over user-recorded samples,
    persisted to a small JSON file so gestures survive restarts.

No GUI code lives here — main.py owns the window, engine.py owns the
camera/landmark loop.
"""

import json
import os
from collections import Counter

import numpy as np

CUSTOM_GESTURES_FILE = "custom_gestures.json"

BUILTIN_WORDS = ["USE", "TECHNOLOGY", "FOR", "HELP", "NOT", "WAR"]

# Landmark indices (MediaPipe hand model, 21 points per hand)
TIP_IDS = [4, 8, 12, 16, 20]      # thumb, index, middle, ring, pinky
PIP_IDS = [3, 6, 10, 14, 18]

# A few FaceLandmarker mesh indices used as rough anchor points
FACE_CHIN = 152
FACE_LEFT_TEMPLE = 127
FACE_RIGHT_TEMPLE = 356


# --------------------------------------------------------------------------
# Feature extraction (shared by builtin + custom classifiers)
# --------------------------------------------------------------------------

def hand_feature_vector(hand_landmarks):
    """63-dim vector: 21 (x,y,z) points, wrist-centered and scale-normalized.

    Making the vector translation/scale invariant means the same gesture
    is recognized regardless of where the hand is in frame or how close
    it is to the camera.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float64)
    pts -= pts[0]  # translate so wrist is the origin
    scale = np.linalg.norm(pts[9])  # distance from wrist to middle-finger MCP
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale
    return pts.flatten()


def two_hand_feature_vector(hand_landmarks_list, handedness_list):
    """Concatenate two single-hand vectors, ordered Left-then-Right so the
    same physical gesture always produces the same vector layout."""
    order = sorted(
        range(len(hand_landmarks_list)),
        key=lambda i: handedness_list[i],
    )
    vecs = [hand_feature_vector(hand_landmarks_list[i]) for i in order]
    return np.concatenate(vecs)


def fingers_extended(hand_landmarks, handedness_label):
    """Returns [thumb, index, middle, ring, pinky] booleans."""
    ext = []
    if handedness_label == "Right":
        ext.append(hand_landmarks[4].x < hand_landmarks[3].x)
    else:
        ext.append(hand_landmarks[4].x > hand_landmarks[3].x)
    for tip, pip in zip(TIP_IDS[1:], PIP_IDS[1:]):
        ext.append(hand_landmarks[tip].y < hand_landmarks[pip].y)
    return ext


def is_fist(fingers):
    return not any(fingers)


def is_open_palm(fingers):
    return all(fingers)


def _dist2d(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


# --------------------------------------------------------------------------
# Built-in rule-based classifier
# --------------------------------------------------------------------------

def classify_builtin(hands, face_landmarks):
    """
    hands: list of dicts, each {"landmarks": [...21 pts...],
                                 "handedness": "Left"/"Right",
                                 "fingers": [thumb,index,middle,ring,pinky]}
    face_landmarks: list of face mesh points, or None.

    Returns (word_or_None, confidence 0..1, both_hands_open bool)
    """
    both_open = False

    if len(hands) == 2:
        f0, f1 = hands[0]["fingers"], hands[1]["fingers"]
        if is_open_palm(f0) and is_open_palm(f1):
            both_open = True

        # WAR: both hands showing index+middle+ring (a "W" handshape)
        w_shape = [False, True, True, True, False]
        score0 = sum(a == b for a, b in zip(f0, w_shape)) / 5
        score1 = sum(a == b for a, b in zip(f1, w_shape)) / 5
        if score0 > 0.8 and score1 > 0.8:
            return "WAR", min(score0, score1), both_open

        # TECHNOLOGY: two closed fists held up side by side (a "T" fist shape)
        if is_fist(f0) and is_fist(f1):
            return "TECHNOLOGY", 0.9, both_open

        # HELP: one hand is a fist-with-thumb-up resting on the other,
        # which is a flat open palm underneath it.
        thumbs_up = [True, False, False, False, False]
        for fist_i, palm_i in ((0, 1), (1, 0)):
            fist_hand, palm_hand = hands[fist_i], hands[palm_i]
            fist_score = sum(a == b for a, b in zip(fist_hand["fingers"], thumbs_up)) / 5
            if fist_score > 0.75 and is_open_palm(palm_hand["fingers"]):
                w0 = fist_hand["landmarks"][0]
                w1 = palm_hand["landmarks"][0]
                if _dist2d(w0, w1) < 0.30:
                    return "HELP", fist_score, both_open

    if len(hands) == 1:
        f = hands[0]["fingers"]
        lm = hands[0]["landmarks"]

        # USE: index + middle extended, others curled ("U" handshape)
        u_shape = [False, True, True, False, False]
        score = sum(a == b for a, b in zip(f, u_shape)) / 5
        if score > 0.8:
            return "USE", score, both_open

        # FOR: only the index finger extended, held near the temple
        point_shape = [False, True, False, False, False]
        score = sum(a == b for a, b in zip(f, point_shape)) / 5
        if score > 0.8 and face_landmarks is not None:
            tip = lm[8]
            temple_dist = min(
                _dist2d(tip, face_landmarks[FACE_LEFT_TEMPLE]),
                _dist2d(tip, face_landmarks[FACE_RIGHT_TEMPLE]),
            )
            if temple_dist < 0.18:
                return "FOR", score, both_open

        # NOT: only the thumb extended ("A" fist with thumb out), near chin
        thumb_shape = [True, False, False, False, False]
        score = sum(a == b for a, b in zip(f, thumb_shape)) / 5
        if score > 0.8 and face_landmarks is not None:
            tip = lm[4]
            chin_dist = _dist2d(tip, face_landmarks[FACE_CHIN])
            if chin_dist < 0.20:
                return "NOT", score, both_open

    return None, 0.0, both_open


# --------------------------------------------------------------------------
# Custom trainable classifier (nearest centroid over recorded samples)
# --------------------------------------------------------------------------

class CustomGestureManager:
    def __init__(self, path=CUSTOM_GESTURES_FILE):
        self.path = path
        self.data = {}       # name -> list of feature vectors (as plain lists)
        self.centroids = {}  # name -> np.ndarray
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.data = {}
        self._rebuild_centroids()

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f)
        except OSError:
            pass

    def add_sample(self, name, feature_vector):
        name = name.strip().upper()
        self.data.setdefault(name, []).append([float(v) for v in feature_vector])

    def finish_recording(self):
        """Call after adding samples for a session to persist + retrain."""
        self._rebuild_centroids()
        self.save()

    def delete_gesture(self, name):
        self.data.pop(name, None)
        self.centroids.pop(name, None)
        self.save()

    def retrain(self):
        self._rebuild_centroids()
        self.save()

    def _rebuild_centroids(self):
        self.centroids = {}
        for name, samples in self.data.items():
            if not samples:
                continue
            lengths = Counter(len(s) for s in samples)
            common_len = lengths.most_common(1)[0][0]
            filtered = [s for s in samples if len(s) == common_len]
            self.centroids[name] = np.mean(np.array(filtered), axis=0)

    def predict(self, feature_vector):
        if not self.centroids:
            return None, 0.0
        fv = np.array(feature_vector)
        best_name, best_dist = None, float("inf")
        for name, centroid in self.centroids.items():
            if centroid.shape[0] != fv.shape[0]:
                continue
            dist = float(np.linalg.norm(fv - centroid))
            if dist < best_dist:
                best_dist, best_name = dist, name
        if best_name is None:
            return None, 0.0
        # Empirical distance-to-confidence scale tuned for normalized
        # wrist-centered landmark vectors.
        confidence = max(0.0, 1.0 - best_dist / 3.5)
        return best_name, confidence

    def list_gestures(self):
        return sorted(self.data.keys())

    def sample_count(self, name):
        return len(self.data.get(name, []))


# --------------------------------------------------------------------------
# Emotion heuristic from FaceLandmarker blendshapes
# --------------------------------------------------------------------------

def top2_emotions(blendshape_categories):
    """blendshape_categories: list of Category(category_name, score).
    Returns [(label, pct), (label, pct)] sorted descending.
    This is a lightweight heuristic combining ARKit-style blendshape
    scores into five coarse emotion buckets — a visual-only extra, not a
    clinically validated emotion classifier.
    """
    scores = {c.category_name: c.score for c in blendshape_categories}

    def g(name):
        return scores.get(name, 0.0)

    smile = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2
    frown = (g("mouthFrownLeft") + g("mouthFrownRight")) / 2
    brow_up = (g("browInnerUp") + g("browOuterUpLeft") + g("browOuterUpRight")) / 3
    brow_down = (g("browDownLeft") + g("browDownRight")) / 2
    jaw_open = g("jawOpen")
    eye_wide = (g("eyeWideLeft") + g("eyeWideRight")) / 2
    squint = (g("eyeSquintLeft") + g("eyeSquintRight")) / 2

    raw = {
        "Happy": smile * 1.4,
        "Surprised": (brow_up * 0.7 + jaw_open * 0.5 + eye_wide * 0.6),
        "Sad": (frown * 1.2 + brow_up * 0.2),
        "Angry": (brow_down * 1.3 + squint * 0.4),
        "Neutral": 0.15,  # small baseline so it can still win when face is calm
    }

    total = sum(raw.values()) or 1e-6
    pct = {k: v / total for k, v in raw.items()}
    ranked = sorted(pct.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:2]

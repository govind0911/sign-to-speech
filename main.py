"""
TSWAG Glasses — real-time sign-to-speech desktop app
======================================================
Watches a webcam, tracks hands + face with MediaPipe, recognizes a small
vocabulary of signs (USE, TECHNOLOGY, FOR, HELP, NOT, WAR + custom
trainable gestures), builds a spoken sentence, and reads it back with
offline text-to-speech.

Run:  python main.py
First run downloads two small MediaPipe model files (~15 MB) and will
install any missing pip packages automatically.
"""

import importlib
import queue
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# Auto-install missing dependencies before anything else imports them
# --------------------------------------------------------------------------
REQUIRED = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "mediapipe": "mediapipe",
    "pyttsx3": "pyttsx3",
    "PyQt6": "PyQt6",
}

def _ensure_packages():
    missing = []
    for module_name, pip_name in REQUIRED.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

_ensure_packages()

# --------------------------------------------------------------------------

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QProgressBar, QPushButton, QDialog, QComboBox, QSlider, QCheckBox,
    QFormLayout, QLineEdit, QListWidget, QMessageBox, QStatusBar,
    QGroupBox, QTextEdit, QListWidgetItem,
)

import engine
import gestures

APP_TITLE = "TSWAG Glasses"

LIGHT_STYLE = """
QWidget { background-color: #f7f9fb; color: #1d1f21; font-family: Segoe UI, sans-serif; }
QMainWindow { background-color: #f7f9fb; }
QGroupBox { border: 1px solid #dfe3e8; border-radius: 8px; margin-top: 10px;
            font-weight: 600; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #1a73e8; }
QPushButton { background-color: #1a73e8; color: white; border: none; border-radius: 6px;
              padding: 8px 16px; font-weight: 600; }
QPushButton:hover { background-color: #1765cc; }
QPushButton:disabled { background-color: #c7d5ea; }
QPushButton#danger { background-color: #e8483a; }
QPushButton#danger:hover { background-color: #cf3a2d; }
QProgressBar { border: 1px solid #dfe3e8; border-radius: 6px; text-align: center;
               background-color: #eef1f5; height: 18px; }
QProgressBar::chunk { background-color: #1a73e8; border-radius: 6px; }
QLineEdit, QComboBox, QListWidget, QTextEdit { border: 1px solid #dfe3e8; border-radius: 6px;
              padding: 4px; background-color: white; }
QLabel#wordLabel { font-size: 26px; font-weight: 700; color: #1a73e8; }
QLabel#wordLabelLocked { font-size: 26px; font-weight: 700; color: #1e8e3e; }
QLabel#sentenceLabel { font-size: 16px; color: #1d1f21; }
QLabel#sectionTitle { font-weight: 700; color: #5f6368; }
"""


# ==========================================================================
# Text-to-speech worker (own thread, blocking pyttsx3 calls stay off the GUI)
# ==========================================================================

class TTSWorker(QThread):
    ready = pyqtSignal(list)   # list of (id, name) voices, once engine is up
    failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._q = queue.Queue()
        self._running = True
        self.voice_id = None

    def run(self):
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        voices = tts_engine.getProperty("voices") or []
        voice_list = [(v.id, v.name) for v in voices]
        female = next((v for v in voices if "female" in (v.name or "").lower()
                       or "female" in " ".join(getattr(v, "gender", []) or []).lower()
                       or "zira" in (v.name or "").lower()), None)
        if female:
            self.voice_id = female.id
            tts_engine.setProperty("voice", female.id)
        self.ready.emit(voice_list)

        while self._running:
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                if self.voice_id:
                    tts_engine.setProperty("voice", self.voice_id)
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception:
                pass

    def set_voice(self, voice_id):
        self.voice_id = voice_id

    def speak(self, text):
        if text:
            self._q.put(text)

    def stop(self):
        self._running = False
        self._q.put(None)
        self.wait(2000)


# ==========================================================================
# Settings dialog
# ==========================================================================

class SettingsDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)

        form = QFormLayout()

        self.camera_combo = QComboBox()
        cams = engine.list_available_cameras()
        for c in cams:
            self.camera_combo.addItem(f"Camera {c}", c)
        idx = self.camera_combo.findData(main_window.engine.camera_index)
        if idx >= 0:
            self.camera_combo.setCurrentIndex(idx)
        form.addRow("Camera:", self.camera_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Built-in signs", "builtin")
        self.mode_combo.addItem("Custom trained gestures", "custom")
        self.mode_combo.setCurrentIndex(0 if main_window.engine.recognition_mode == "builtin" else 1)
        form.addRow("Recognition mode:", self.mode_combo)

        self.voice_combo = QComboBox()
        for vid, name in main_window.available_voices:
            self.voice_combo.addItem(name, vid)
        current_idx = self.voice_combo.findData(main_window.tts.voice_id)
        if current_idx >= 0:
            self.voice_combo.setCurrentIndex(current_idx)
        form.addRow("Voice:", self.voice_combo)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(50, 95)
        self.conf_slider.setValue(int(main_window.engine.confidence_threshold * 100))
        self.conf_value_label = QLabel(f"{self.conf_slider.value()}%")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value_label.setText(f"{v}%"))
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_value_label)
        form.addRow("Confidence threshold:", conf_row)

        self.hold_slider = QSlider(Qt.Orientation.Horizontal)
        self.hold_slider.setRange(5, 30)  # tenths of a second: 0.5s - 3.0s
        self.hold_slider.setValue(int(main_window.hold_duration * 10))
        self.hold_value_label = QLabel(f"{self.hold_slider.value()/10:.1f}s")
        self.hold_slider.valueChanged.connect(
            lambda v: self.hold_value_label.setText(f"{v/10:.1f}s"))
        hold_row = QHBoxLayout()
        hold_row.addWidget(self.hold_slider)
        hold_row.addWidget(self.hold_value_label)
        form.addRow("Hold duration:", hold_row)

        self.emotion_check = QCheckBox("Enable emotion detection")
        self.emotion_check.setChecked(main_window.engine.emotion_enabled)
        form.addRow(self.emotion_check)

        self.speech_check = QCheckBox("Enable speech output")
        self.speech_check.setChecked(main_window.speech_enabled)
        form.addRow(self.speech_check)

        self.fullscreen_button = QPushButton("Toggle Fullscreen")
        self.fullscreen_button.clicked.connect(main_window.toggle_fullscreen)
        form.addRow(self.fullscreen_button)

        buttons = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_and_close)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color:#9aa0a6;")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(apply_btn)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def apply_and_close(self):
        mw = self.main_window
        new_cam = self.camera_combo.currentData()
        cam_changed = new_cam != mw.engine.camera_index
        mw.engine.recognition_mode = self.mode_combo.currentData()
        mw.engine.confidence_threshold = self.conf_slider.value() / 100.0
        mw.engine.emotion_enabled = self.emotion_check.isChecked()
        mw.hold_duration = self.hold_slider.value() / 10.0
        mw.speech_enabled = self.speech_check.isChecked()
        mw.tts.set_voice(self.voice_combo.currentData())
        if cam_changed:
            mw.restart_camera(new_cam)
        self.accept()


# ==========================================================================
# Gesture trainer dialog
# ==========================================================================

class TrainerDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Custom Gesture Trainer")
        self.setMinimumWidth(420)

        layout = QVBoxLayout()

        info = QLabel("Hold a one- or two-hand pose steady in front of the camera, "
                       "then press Record. Recording captures ~2 seconds of samples.")
        info.setWordWrap(True)
        layout.addWidget(info)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Gesture name:"))
        self.name_edit = QLineEdit()
        name_row.addWidget(self.name_edit)
        layout.addLayout(name_row)

        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.start_record)
        layout.addWidget(self.record_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 30)
        layout.addWidget(self.progress)

        layout.addWidget(QLabel("Saved gestures:"))
        self.list_widget = QListWidget()
        self._refresh_list()
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        retrain_btn = QPushButton("Retrain")
        retrain_btn.clicked.connect(self.retrain)
        delete_btn = QPushButton("Delete Selected")
        delete_btn.setObjectName("danger")
        delete_btn.clicked.connect(self.delete_selected)
        btn_row.addWidget(retrain_btn)
        btn_row.addWidget(delete_btn)
        layout.addLayout(btn_row)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

        main_window.engine.recording_progress.connect(self.on_progress)
        main_window.engine.recording_done.connect(self.on_done)

    def _refresh_list(self):
        self.list_widget.clear()
        mgr = self.main_window.custom_manager
        for name in mgr.list_gestures():
            self.list_widget.addItem(QListWidgetItem(f"{name}  ({mgr.sample_count(name)} samples)"))

    def start_record(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Name required", "Type a gesture name first.")
            return
        self.record_btn.setEnabled(False)
        self.progress.setValue(0)
        self.main_window.engine.start_recording_gesture(name, num_frames=30)

    def on_progress(self, frames_left):
        self.progress.setValue(30 - frames_left)

    def on_done(self, name):
        self.record_btn.setEnabled(True)
        self._refresh_list()
        QMessageBox.information(self, "Recorded", f"Saved samples for '{name}'.")

    def retrain(self):
        self.main_window.custom_manager.retrain()
        self._refresh_list()
        QMessageBox.information(self, "Retrained", "Custom gesture classifier retrained.")

    def delete_selected(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        name = item.text().split(" ")[0]
        self.main_window.custom_manager.delete_gesture(name)
        self._refresh_list()


# ==========================================================================
# Main window
# ==========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1080, 680)

        self.custom_manager = gestures.CustomGestureManager()
        self.engine = engine.CameraEngine(self.custom_manager)
        self.available_voices = []

        self.tts = TTSWorker()
        self.tts.ready.connect(self.on_voices_ready)
        self.tts.failed.connect(lambda msg: self.status_bar.showMessage(
            f"Voice output unavailable: {msg}"))
        self.tts.start()

        # runtime state
        self.hold_duration = 1.5
        self.speech_enabled = True
        self.current_candidate = None
        self.candidate_start_time = None
        self.locked_this_hold = False
        self.unmatched_start_time = None
        self.said_unknown = False
        self.sentence_tokens = []
        self.seen_words = set()
        self.sentence_history = []
        self.both_open_start = None
        self.asked_this_open_hold = False
        self.is_fullscreen = False

        self._build_ui()
        self._build_menu()

        self.engine.frame_ready.connect(self.on_frame)
        self.engine.error.connect(self.on_error)
        self.engine.status.connect(self.status_bar.showMessage)
        self.engine.start()

    # ---------------------------------------------------------------- UI --

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        # Left: video + recognition state
        left = QVBoxLayout()
        self.video_label = QLabel("Starting camera ...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color:#000; border-radius:8px; color:white;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.video_label)

        self.word_label = QLabel("—")
        self.word_label.setObjectName("wordLabel")
        self.word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.word_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        left.addWidget(self.progress_bar)

        stats_row = QHBoxLayout()
        self.fps_label = QLabel("FPS: —")
        stats_row.addWidget(self.fps_label)
        stats_row.addStretch()
        left.addLayout(stats_row)

        root.addLayout(left, 3)

        # Right: sentence, emotion, controls
        right = QVBoxLayout()

        sentence_box = QGroupBox("Sentence Builder")
        sb_layout = QVBoxLayout()
        self.sentence_label = QLabel("")
        self.sentence_label.setObjectName("sentenceLabel")
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setMinimumHeight(80)
        sb_layout.addWidget(self.sentence_label)
        sb_btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.clear_sentence(add_to_history=True))
        read_btn = QPushButton("Read Aloud")
        read_btn.clicked.connect(self.prompt_read_sentence)
        sb_btn_row.addWidget(clear_btn)
        sb_btn_row.addWidget(read_btn)
        sb_layout.addLayout(sb_btn_row)
        sentence_box.setLayout(sb_layout)
        right.addWidget(sentence_box)

        emotion_box = QGroupBox("Emotion (visual only)")
        em_layout = QVBoxLayout()
        self.emotion_label = QLabel("—")
        em_layout.addWidget(self.emotion_label)
        emotion_box.setLayout(em_layout)
        right.addWidget(emotion_box)

        history_box = QGroupBox("Sentence History (last 10)")
        hist_layout = QVBoxLayout()
        self.history_list = QListWidget()
        hist_layout.addWidget(self.history_list)
        history_box.setLayout(hist_layout)
        right.addWidget(history_box, 1)

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.open_settings)
        trainer_btn = QPushButton("Gesture Trainer")
        trainer_btn.clicked.connect(self.open_trainer)
        right.addWidget(settings_btn)
        right.addWidget(trainer_btn)

        root.addLayout(right, 2)

        self.setCentralWidget(central)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.setStyleSheet(LIGHT_STYLE)

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")

        settings_act = QAction("Settings", self)
        settings_act.triggered.connect(self.open_settings)
        menu.addAction(settings_act)

        trainer_act = QAction("Gesture Trainer", self)
        trainer_act.triggered.connect(self.open_trainer)
        menu.addAction(trainer_act)

        fullscreen_act = QAction("Toggle Fullscreen", self)
        fullscreen_act.triggered.connect(self.toggle_fullscreen)
        menu.addAction(fullscreen_act)

        menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        menu.addAction(exit_act)

    # ------------------------------------------------------------ voices --

    def on_voices_ready(self, voice_list):
        self.available_voices = voice_list

    # ------------------------------------------------------------- frame --

    def on_frame(self, data):
        pix = QPixmap.fromImage(data["image"]).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pix)
        self.fps_label.setText(f"FPS: {data['fps']:.0f}")

        if data["emotions"]:
            lines = [f"{name} — {pct*100:.0f}%" for name, pct in data["emotions"]]
            self.emotion_label.setText("\n".join(lines))
        else:
            self.emotion_label.setText("—")

        self._handle_recognition(data)
        self._handle_read_trigger(data)

    def _handle_recognition(self, data):
        word = data["word"]
        now = time.time()

        if word is None:
            self.current_candidate = None
            self.candidate_start_time = None
            self.locked_this_hold = False
            self.progress_bar.setValue(0)

            if data["num_hands"] > 0:
                if self.unmatched_start_time is None:
                    self.unmatched_start_time = now
                    self.said_unknown = False
                elapsed = now - self.unmatched_start_time
                if elapsed < 3.0:
                    self.word_label.setObjectName("wordLabel")
                    self.word_label.setStyleSheet("")
                    self.word_label.setText("Not recognized. Try again.")
                else:
                    self.word_label.setText("Unknown")
                    if not self.said_unknown:
                        self.said_unknown = True
                        if self.speech_enabled:
                            self.tts.speak("Unknown")
            else:
                self.unmatched_start_time = None
                self.word_label.setText("—")
            return

        self.unmatched_start_time = None

        if word == self.current_candidate:
            elapsed = now - self.candidate_start_time
            progress = min(1.0, elapsed / self.hold_duration)
            self.progress_bar.setValue(int(progress * 100))
            if elapsed >= self.hold_duration and not self.locked_this_hold:
                self.lock_word(word)
                self.locked_this_hold = True
        else:
            self.current_candidate = word
            self.candidate_start_time = now
            self.locked_this_hold = False
            self.progress_bar.setValue(0)
            self.word_label.setObjectName("wordLabel")
            self.word_label.setStyleSheet("color:#1a73e8;")
            self.word_label.setText(word)

    def lock_word(self, word):
        display = word if word not in self.seen_words else f"{word}!!!"
        self.seen_words.add(word)
        self.sentence_tokens.append(display)
        self.sentence_label.setText(" ".join(self.sentence_tokens))

        self.word_label.setStyleSheet("color:#1e8e3e;")
        self.word_label.setText(f"{display}  ✓")
        QTimer.singleShot(900, lambda: self.word_label.setStyleSheet("color:#1a73e8;"))

        self.status_bar.showMessage(f"Locked: {display}", 2000)
        if self.speech_enabled:
            self.tts.speak(word)

    def _handle_read_trigger(self, data):
        now = time.time()
        if data["both_hands_open"]:
            if self.both_open_start is None:
                self.both_open_start = now
            elapsed = now - self.both_open_start
            if elapsed >= 2.0 and not self.asked_this_open_hold:
                self.asked_this_open_hold = True
                self.prompt_read_sentence()
        else:
            self.both_open_start = None
            self.asked_this_open_hold = False

    # --------------------------------------------------------- sentence --

    def prompt_read_sentence(self):
        if not self.sentence_tokens:
            QMessageBox.information(self, "Nothing to read", "The sentence is currently empty.")
            return
        text = " ".join(self.sentence_tokens)
        reply = QMessageBox.question(
            self, "Read Sentence?", f'Read this aloud?\n\n"{text}"',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.speech_enabled:
                self.tts.speak(text.replace("!!!", ""))
            clear_reply = QMessageBox.question(
                self, "Clear Sentence?", "Clear the sentence now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if clear_reply == QMessageBox.StandardButton.Yes:
                self.clear_sentence(add_to_history=True)

    def clear_sentence(self, add_to_history=False):
        if add_to_history and self.sentence_tokens:
            text = " ".join(self.sentence_tokens)
            self.sentence_history.insert(0, text)
            self.sentence_history = self.sentence_history[:10]
            self.history_list.clear()
            self.history_list.addItems(self.sentence_history)
        self.sentence_tokens = []
        self.seen_words = set()
        self.sentence_label.setText("")

    # ---------------------------------------------------------- dialogs --

    def open_settings(self):
        SettingsDialog(self).exec()

    def open_trainer(self):
        TrainerDialog(self).exec()

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self.is_fullscreen = not self.is_fullscreen

    # ----------------------------------------------------------- camera --

    def restart_camera(self, new_index):
        self.engine.stop()
        self.engine = engine.CameraEngine(self.custom_manager)
        self.engine.camera_index = new_index
        self.engine.frame_ready.connect(self.on_frame)
        self.engine.error.connect(self.on_error)
        self.engine.status.connect(self.status_bar.showMessage)
        self.engine.start()

    def on_error(self, message):
        self.status_bar.showMessage(message, 5000)
        QMessageBox.warning(self, "TSWAG Glasses", message)

    # ------------------------------------------------------------- exit --

    def closeEvent(self, event):
        self.engine.stop()
        self.tts.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

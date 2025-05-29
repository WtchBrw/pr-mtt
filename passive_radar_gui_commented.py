# Passive Radar GUI with Multi-Antenna Support, Tracking, and Real-Time Logging
# Author: NotYourFathersLore
# Description: This script implements a passive radar using RTL-SDR with real-time GUI, plotting,
#              range-Doppler processing, target clustering, tracking, and SQLite logging.

import sys
import time
import numpy as np
import sqlite3
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QLabel, QPushButton, QTabWidget, QCheckBox, QHBoxLayout)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from rtlsdr import RtlSdr
from scipy.signal import correlate
from scipy.fft import fft, fftshift
from sklearn.cluster import DBSCAN
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Constants
SAMPLE_RATE = 2.4e6
CENTER_FREQ = 100.1e6
GAIN = 'auto'
SAMPLES = 1024 * 64
WINDOW_SIZE = 256
OVERLAP = 0.5
C = 3e8  # Speed of light
FREQ = CENTER_FREQ
WAVELENGTH = C / FREQ
UPDATE_INTERVAL = 2000  # milliseconds

# Set up a single reference SDR
ref = RtlSdr()
ref.sample_rate = SAMPLE_RATE
ref.center_freq = CENTER_FREQ
ref.gain = GAIN

# SQLite database setup
conn = sqlite3.connect("radar_tracks.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS tracks (
        id TEXT,
        antenna INTEGER,
        range REAL,
        velocity REAL,
        timestamp TEXT
    )
''')
conn.commit()

# Simple Target Tracker Class
class SimpleTracker:
    def __init__(self, max_lost=5):
        self.next_id = 0
        self.tracks = {}
        self.max_lost = max_lost

    def update(self, detections):
        updated_tracks = {}
        for det in detections:
            matched = False
            for tid, (prev_pos, lost) in self.tracks.items():
                if np.linalg.norm(np.array(prev_pos) - np.array(det)) < 5:
                    updated_tracks[tid] = (det, 0)
                    matched = True
                    break
            if not matched:
                updated_tracks[self.next_id] = (det, 0)
                self.next_id += 1

        for tid in self.tracks:
            if tid not in updated_tracks:
                pos, lost = self.tracks[tid]
                if lost + 1 < self.max_lost:
                    updated_tracks[tid] = (pos, lost + 1)

        self.tracks = updated_tracks
        return self.tracks

# Radar Processing Worker Thread
class RadarWorker(QThread):
    processed_data_ready = pyqtSignal(int, np.ndarray, list)

    def __init__(self, antenna_index, sdr_ref, sdr_surv):
        super().__init__()
        self.antenna_index = antenna_index
        self.sdr_ref = sdr_ref
        self.sdr_surv = sdr_surv
        self.running = True
        self.tracker = SimpleTracker()

    def run(self):
        while self.running:
            try:
                ref = self.sdr_ref.read_samples(SAMPLES)
                surv = self.sdr_surv.read_samples(SAMPLES)
            except Exception:
                continue

            step = int(WINDOW_SIZE * (1 - OVERLAP))
            n_windows = (len(surv) - WINDOW_SIZE) // step
            rd_matrix = []

            for i in range(n_windows):
                start = i * step
                end = start + WINDOW_SIZE
                corr = correlate(surv[start:end], ref[start:end], mode='full')
                corr_mag = np.abs(corr)
                rd_matrix.append(corr_mag)

            if not rd_matrix:
                continue

            rd_matrix = np.array(rd_matrix).T
            rd_fft = np.abs(fftshift(fft(rd_matrix, axis=1), axes=1))
            rd_db = 20 * np.log10(rd_fft + 1e-6)

            threshold_db = np.max(rd_db) - 10
            peaks_y, peaks_x = np.where(rd_db > threshold_db)
            if len(peaks_x) > 0:
                coords = np.stack((peaks_x, peaks_y), axis=1)
                clusters = DBSCAN(eps=3, min_samples=2).fit(coords)
                clustered_coords = []
                for label in set(clusters.labels_):
                    if label == -1:
                        continue
                    members = coords[clusters.labels_ == label]
                    centroid = np.mean(members, axis=0)
                    clustered_coords.append(tuple(centroid))

                tracked = self.tracker.update(clustered_coords)
                target_info = []
                for tid, (pos, _) in tracked.items():
                    rb = pos[0] * C / (2 * SAMPLE_RATE)
                    vb = (pos[1] - rd_db.shape[0] / 2) * (SAMPLE_RATE / rd_db.shape[0]) * WAVELENGTH / 2
                    target_info.append((tid, rb, vb))
                    cursor.execute("INSERT INTO tracks VALUES (?, ?, ?, ?, datetime('now'))",
                                   (str(tid), self.antenna_index, rb, vb))
                    conn.commit()

                self.processed_data_ready.emit(self.antenna_index, rd_db, target_info)

            time.sleep(UPDATE_INTERVAL / 1000)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# GUI Class
class PassiveRadarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Passive Radar with Multi-Antenna Tracking")
        self.setGeometry(100, 100, 1200, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.control_panel = QHBoxLayout()
        self.start_button = QPushButton("Start Radar")
        self.start_button.clicked.connect(self.toggle_radar)
        self.control_panel.addWidget(self.start_button)

        self.antenna_checkboxes = []

        # THIS FOR-LOOP SETS NUMBER OF SDR PANELS AND SDRs
        for i in range(4):
            checkbox = QCheckBox(f"Antenna {i+1}")
            checkbox.setChecked(i == 0)
            self.control_panel.addWidget(checkbox)
            self.antenna_checkboxes.append(checkbox)

        self.layout.addLayout(self.control_panel)

        self.tabs = QTabWidget()
        self.plots = []
        for i in range(4):
            fig, ax = plt.subplots()
            canvas = FigureCanvas(fig)
            self.plots.append((fig, ax, canvas))
            tab = QWidget()
            tab_layout = QVBoxLayout()
            tab_layout.addWidget(canvas)
            tab.setLayout(tab_layout)
            self.tabs.addTab(tab, f"Antenna {i+1}")
        self.layout.addWidget(self.tabs)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.workers = []
        self.running = False

    def toggle_radar(self):
        if not self.running:
            self.running = True
            self.start_button.setText("Stop Radar")

            #COMMENT THIS OUT TO CHECK GUI LAYOUT W/O SDR RUNNING
            self.start_workers()
        else:
            self.running = False
            self.start_button.setText("Start Radar")
            self.stop_workers()

    def start_workers(self):
        self.workers = []

        for i, checkbox in enumerate(self.antenna_checkboxes):
            if checkbox.isChecked():
                if i == 0:
                    continue  # Skip if this is the reference antenna (reference is always zero)

                #set sdr interface to match the number of the box that is checked
                surv = RtlSdr()
                surv.sample_rate = SAMPLE_RATE
                surv.center_freq = CENTER_FREQ
                surv.gain = GAIN

                worker = RadarWorker(i, ref, surv)
                worker.processed_data_ready.connect(self.handle_processed_data)
                worker.start()
                self.workers.append(worker)

    def stop_workers(self):
        for worker in self.workers:
            worker.stop()
        self.workers.clear()

    def handle_processed_data(self, antenna_index, rd_db, targets):
        fig, ax, canvas = self.plots[antenna_index]
        ax.clear()
        ax.imshow(rd_db, aspect='auto', origin='lower', extent=[0, rd_db.shape[1], 0, rd_db.shape[0]])
        for tid, rb, vb in targets:
            ax.plot(rb, vb, 'ro')
            ax.text(rb, vb, f'ID {tid}', color='white', fontsize=8)
        ax.set_title(f"Antenna {antenna_index+1} Range-Doppler")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Speed (m/s)")
        canvas.draw()

    def update_gui(self):
        pass  # Placeholder for future updates

    def closeEvent(self, event):
        self.stop_workers()
        conn.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    radar_gui = PassiveRadarGUI()
    radar_gui.show()
    sys.exit(app.exec_())

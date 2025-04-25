# Passive Radar GUI with Multi-Antenna Support, Tracking, and Real-Time Logging
# Author: [Your Name]
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

from PyQt5.QtWebEngineWidgets import QWebEngineView
import folium
import io

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
NUM_SDR_ANTENNAS = 5
REF_SDR_SERIAL = 1000

# Set up a single reference SDR
ref = RtlSdr(serial_number=str(REF_SDR_SERIAL))
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

# Map Tab
class MapTab(QWidget):
    def __init__(self, radar_lat=43.43577769387483, radar_lon=-116.2726627579046):
        super().__init__()
        self.radar_lat = radar_lat
        self.radar_lon = radar_lon
        self.target_markers = {}
        self.layout = QVBoxLayout(self)
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)
        self.update_map([])

    def update_map(self, targets):
        m = folium.Map(location=[self.radar_lat, self.radar_lon], zoom_start=14, tiles='Esri.WorldImagery')

        folium.Marker(
            [self.radar_lat, self.radar_lon],
            tooltip="Radar Location",
            icon=folium.Icon(color='red', icon='wifi')
        ).add_to(m)

        for tid, lat, lon in targets:
            folium.Marker(
                [lat, lon],
                tooltip=f"ID: {tid}\nLat: {lat:.5f}\nLon: {lon:.5f}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

        data = io.BytesIO()
        m.save(data, close_file=False)
        self.web_view.setHtml(data.getvalue().decode())
        
    def update_targets(self, target_list):
        self.update_map(target_list)


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

        for i in range(NUM_SDR_ANTENNAS):
            print(i)
            checkbox = QCheckBox(f"Antenna {i+1}")
            checkbox.setChecked(i == 0)
            self.control_panel.addWidget(checkbox)
            self.antenna_checkboxes.append(checkbox)

        self.layout.addLayout(self.control_panel)

        # THE LAT AND LONG HERE ANCHOR THE RADAR TO A MAP LOCATION
        self.tabs = QTabWidget()
        self.map_tab = MapTab(radar_lat=43.43577769387483, radar_lon=-116.2726627579046)
        self.tabs.addTab(self.map_tab, "Target Map")
        self.latest_targets = []

        self.plots = []
        for i in range(NUM_SDR_ANTENNAS):
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
                surv = RtlSdr(serial_number=str(REF_SDR_SERIAL + i))
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

        mapped_targets = []
        for tid, rb, vb in targets:
            ax.plot(rb, vb, 'ro')
            ax.text(rb, vb, f'ID {tid}', color='white', fontsize=8)

            # Estimate lat/lon shift (rough)
            dlat = (rb / 111320)  # meters to degrees latitude
            dlon = (rb / (40075000 * np.cos(np.radians(self.map_tab.radar_lat)) / 360))  # meters to degrees longitude
            mapped_targets.append((tid, self.map_tab.radar_lat + dlat, self.map_tab.radar_lon + dlon))

        self.latest_targets = mapped_targets
        self.map_tab.update_targets(mapped_targets)

        ax.set_title(f"Antenna {antenna_index+1} Range-Doppler")
        ax.set_xlabel("Range Bin")
        ax.set_ylabel("Doppler Bin")
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

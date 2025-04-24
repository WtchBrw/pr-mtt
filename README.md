# 🛰️ Passive Radar System with Multi-Antenna Support, Real-Time Tracking, and Interactive Mapping

## Overview

This project implements a real-time passive radar system using RTL-SDR devices and Python. It includes:

- Multi-antenna support for enhanced detection coverage.
- Range-Doppler signal processing and clustering for target detection.
- Real-time target tracking and logging to an SQLite database.
- A PyQt5 GUI with:
  - Live range-Doppler plots for each antenna.
  - An interactive satellite-style map showing target positions relative to radar location.

---

## Features

- 📡 **Multi-Antenna Surveillance**  
  Supports multiple RTL-SDR devices for simultaneous surveillance and reference signal capture.

- 📊 **Real-Time Range-Doppler Visualization**  
  Displays live FFT-processed heatmaps per antenna.

- 🎯 **Target Detection & Tracking**  
  Uses correlation, clustering (DBSCAN), and simple ID-based tracking to follow moving targets.

- 🗺️ **Interactive Mapping**  
  Displays tracked targets on a live map using `folium` and `QWebEngineView`:
  - Zoom/pan capabilities.
  - Clickable target markers with metadata.
  - Satellite basemap (Esri imagery).

- 🧠 **SQLite Logging**  
  Logs all target detections into `radar_tracks.db` for later analysis.

---

## Requirements

- Python 3.8+
- At least two RTL-SDR devices (1 reference + 1 or more surveillance antennas)

### Python Dependencies

Install with:

```bash
pip install -r requirements.txt
```

## Notes

- Target latitude and longitude are estimated from range data; accuracy will vary.
- The system assumes line-of-sight and flat-earth approximations for mapping.
- This is a proof-of-concept and educational tool, not a certified radar system.

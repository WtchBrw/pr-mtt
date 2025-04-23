
# Passive Radar GUI - Documentation

This project implements a **real-time passive radar system** using the RTL-SDR platform and Python. The radar passively receives signals (e.g., from FM broadcast stations) and computes target motion using range-Doppler analysis. The application is visualized through a PyQt5 GUI and uses multithreading, real-world units, multi-antenna support, target tracking, clustering, and real-time database logging.

---

## Features & Objective Implementation

### ✅ 1. **Multithreaded Radar Processing**
- **Why:** Prevents GUI freezing while SDR samples are processed.
- **How:** 
  - Each active surveillance antenna is assigned a `RadarWorker` thread.
  - These threads run signal processing routines (correlation, FFT, range-Doppler mapping) asynchronously.
  - Results are emitted via PyQt signals and handled in the main thread for GUI-safe updates.

### ✅ 2. **Multiple Surveillance Antennas (1–4)**
- **Why:** Allows comparison between receivers, directional tracking, or spatial diversity.
- **How:** 
  - Up to 4 antennas are supported.
  - Each has a checkbox in the GUI to toggle it on/off.
  - Each active antenna creates its own `RtlSdr` object and thread.

### ✅ 3. **Tabbed GUI with Real-Time Plots**
- **Why:** Organizes visualization per antenna for clarity and scalability.
- **How:** 
  - A `QTabWidget` holds tabs labeled "Antenna 1" through "Antenna 4".
  - Each tab contains a Matplotlib canvas showing a range-Doppler map.

### ✅ 4. **Range-Doppler Processing**
- **Why:** Enables detection of objects based on delay (range) and Doppler shift (velocity).
- **How:**
  - Cross-correlation is used to find range.
  - Doppler FFT is applied across a sliding window of correlated samples.
  - Units:
    - Range axis in **meters**
    - Velocity axis in **meters per second (m/s)**

### ✅ 5. **Clustering with DBSCAN**
- **Why:** Groups peaks in the range-Doppler map into targets, reducing noise and improving tracking.
- **How:**
  - Peaks are identified where power is within 10 dB of max.
  - `DBSCAN` is applied to spatially cluster these into grouped detections.
  - Centroids of these clusters are passed to the tracker.

### ✅ 6. **Multi-Target Tracking**
- **Why:** Maintains continuity of moving objects and assigns persistent identifiers.
- **How:**
  - `SimpleTracker` maintains a list of `tracks` with positions, history, and a unique ID.
  - Tracks are updated frame-by-frame based on proximity to current detections.
  - Tracks that disappear for a defined number of cycles (`max_lost`) are pruned.

### ✅ 7. **Physical Units & Axes**
- **Why:** Translates sample data into real-world interpretable values.
- **How:**
  - Range in meters using: `range_bin * c / (2 * sample_rate)`
  - Velocity in m/s using: `doppler_freq * λ / 2` (λ = wavelength)

### ✅ 8. **Unique Target Identifiers**
- **Why:** Enables persistent identification across updates and sessions.
- **How:**
  - Each new target is assigned an `id` when initialized.
  - The ID is displayed next to the target in the plot and stored in the database.

### ✅ 9. **Real-Time SQLite Logging**
- **Why:** Keeps a persistent history of target activity for analysis or export.
- **How:**
  - A SQLite database is initialized (`radar_tracks.db`) with a `tracks` table.
  - Each tracked object (range, velocity, time, antenna, ID) is logged in real time using `INSERT INTO`.

---

## File Structure

- `passive_radar_gui.py`: Main GUI + processing logic

---

## Requirements

- Python 3.8+
- RTL-SDR compatible hardware
- Dependencies:
  - `PyQt5`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pyrtlsdr`

```bash
pip install PyQt5 numpy scipy matplotlib scikit-learn pyrtlsdr
```

---

## Running the App

```bash
python passive_radar_gui.py
```

---

## Next Steps (Suggestions)
- Add heading estimation
- Add 2D map plotting
- Desktop alerts for targets of interest
- Target highlighting (show ID on hover)
- Advanced tracking (Kalman filter, JPDA, CFAR etc.)
- Web API for scraping data

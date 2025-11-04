# **People Counting and Direction Detection System**

## **Overview**

This application performs **person detection, tracking, and directional counting (IN/OUT)** using a **video feed**.
It is designed for **entrance/exit monitoring**, such as tracking how many people enter or leave a building.

The system integrates:

* **YOLO-based person detection**
* **BotSort tracking**
* A **curve-based boundary** (the virtual counting line)
* **Automatic orientation detection**
* Persistent **diagnostics and camera orientation**

---

## **System Architecture**

| Component               | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| **YOLO model**          | Detects people (class 0).                                        |
| **BotSort tracker**     | Assigns unique IDs and maintains motion continuity.              |
| **CurveManager**        | Manages curve configuration, orientation, and persistence.       |
| **Orientation Module**  | Determines whether movement is toward or away from the entrance. |
| **Tracker Logic**       | Updates IN/OUT counts based on curve crossings.                  |
| **Visualization Tools** | Optional plotting and debugging for trajectories.                |

---

## **Default Setup and Core Logic**

### **Camera Orientation**

* Default: **camera faces the entrance** (inside the building).
* The user draws a **curve** across the entrance area.
* The system then counts:

  * **Toward the camera → IN** (entering)
  * **Away from the camera → OUT** (exiting)

> **Summary:**
>
> * **IN = toward the camera**
> * **OUT = away from the camera**

---

## **Illustrative Setup (Top-Down View)**

```
top of image
↑
|
|    (461,366)             (741,389)
|       \                       /
|        \                     /
|         \                   /
|          \                 /
|           (555,418)---(674,438)---(721,421)
|
+----------------------------------------→ x
```

* **Y-axis increases downward** (OpenCV coordinate system).
* The **curve** spans the doorway region.
* People walking **upward (toward camera)** → **IN**.
* People walking **downward (away from camera)** → **OUT**.

---

## **Camera Orientation Conventions**

| Camera Orientation | IN_direction (relative to curve) | Notes                                                |
| ------------------ | -------------------------------- | ---------------------------------------------------- |
| Front-facing       | `toward_cam`                     | Camera faces entrance; movement toward camera = IN.  |
| Back-facing        | `away_from_cam`                  | Camera behind crowd; movement away from camera = IN. |
| Left-side          | `left`                           | IN = motion from left side of curve.                 |
| Right-side         | `right`                          | IN = motion from right side of curve.                |
| Overhead           | `left` / `right`                 | Chosen manually; consistent across views.            |

> **Principle:**
> “IN” always means **toward the entrance**, not necessarily toward the camera.

---

## **Curve Management and Configuration Persistence**

The **`CurveManager`** (`modules/curve_manager.py`) handles curve setup and persistence.

| Function                                | Description                                                                       |
| --------------------------------------- | --------------------------------------------------------------------------------- |
| `load_curve_config()`                   | Loads curve points, orientation, and diagnostics from `config/curve_config.json`. |
| `create_curve(frame)`                   | Interactive mode — user draws curve directly on frame if no config is found.      |
| `determine_orientation(sample_anchors)` | Automatically determines orientation when `IN_direction` is `"auto"`.             |
| `save_curve_config(data)`               | Persists curve points, camera orientation, and diagnostics for reuse.             |

### **Automatic Handling Flow**

1. On startup, the system checks if `curve_config.json` exists.
2. If **missing**, the user is prompted to draw a curve interactively.
3. The curve is saved with:

   ```json
   { "IN_direction": "auto" }
   ```
4. When people start moving, the system collects sample points.
5. Once enough samples are collected, **`auto_orient_curve()`** detects orientation.
6. The result, along with detailed diagnostics, is saved back to `curve_config.json`.

---

## **Auto Orientation and Diagnostics**

When `IN_direction` is `"auto"`, the system calls:

```python
auto_orient_curve(curve_points, sample_anchors, eps=3.0, min_crossings=1)
```

This computes:

* The likely direction of **IN vs OUT**
* A set of metrics showing how the decision was made

Example diagnostic output (persisted in config):

```json
{
  "num_samples": 15,
  "num_crossings": 1,
  "crossings": [[0, 14, 1.0, -1.0, 114.93]],
  "candidates": {"+1": 0, "-1": 114.93},
  "orientation": -1,
  "camera_convention": "facing_entrance"
}
```

These diagnostics are saved to `curve_config.json`:

```json
{
  "curve_points": [[461,366], [555,418], [674,438], [721,421]],
  "IN_direction": "auto",
  "orientation": -1,
  "camera_orientation": "facing_entrance",
  "orientation_diagnostics": {
    "num_samples": 15,
    "num_crossings": 1,
    "orientation": -1
  }
}
```

On the next run, orientation is reused — no recalibration needed.

---

## **Counting Mechanism**

1. **Curve Definition**
   User defines the curve via `CurveManager.create_curve()` or loads from config.

2. **Signed Distance Calculation**
   Each tracked person’s anchor point (bottom of bounding box) is compared to the curve:

   ```python
   signed_distance_to_curve(point, curve)
   ```

3. **Orientation Application**
   Signed distances are multiplied by the curve orientation (`+1` or `-1`).

4. **Counting**
   When a track crosses from one side of the curve to the other, `update_track_state()` increments the **IN** or **OUT** counter.

---

## **Running the System**

### **Live or Recorded Video**

```bash
python main.py
```

* Draw the curve when prompted (if config missing)
* The system will:

  * Detect and track people
  * Automatically learn the correct IN/OUT orientation
  * Save results in `config/curve_config.json`

---

## **Offline Testing (Trajectory Mode)**

Run this mode to test orientation detection and labeling logic without video:

```bash
python test_trajectory_orientation.py
```

This script:

* Loads the saved curve configuration
* Simulates sample trajectories
* Evaluates `auto_orient_curve`
* Prints raw distances and IN/OUT classification

---

## **Key Modules**

| Module                           | Description                                            |
| -------------------------------- | ------------------------------------------------------ |
| `modules/curve_manager.py`       | Curve creation, persistence, and diagnostics handling. |
| `modules/orientation.py`         | Auto orientation logic and diagnostic computation.     |
| `modules/tracker_logic.py`       | Curve crossing and IN/OUT counting logic.              |
| `utils/viz_tools.py`             | Visualization and trajectory plotting.                 |
| `main.py`                        | Entry point for real-time counting.                    |
| `test_trajectory_orientation.py` | Offline diagnostic testing for curve orientation.      |

---

## **Parked / To-Revisit Items**

1. **Gray-zone hysteresis**
   Add tolerance near the curve to avoid double counts.

2. **Dynamic camera flip detection**
   Auto-detect when the camera’s facing direction changes.

3. **Real-world mapping**
   Use homography to project image coordinates into real-world distances.

4. **Diagnostics visualization overlay**
   Display diagnostic values directly in live video for debugging.

---

## **Next Steps**

* ✅ Validate `orientation_diagnostics` persistence and reuse
* 🧭 Add live visualization for IN/OUT classification
* ⚙️ Tune `auto_orient_curve` parameters for stability
* 📊 Extend `viz_tools` to overlay curve and IN/OUT sides on video feed

# **People Counting and Direction Detection System**

## **Overview**

This application performs **person detection, tracking, and directional counting (IN/OUT)** from a **video feed**.
It is designed for **entrance/exit monitoring**, for example, counting how many people enter or leave a building through a main doorway.

The system combines:

* YOLO-based person detection
* A tracker (BotSort)
* A geometric **curve-based boundary**
* **Trajectory analysis** with **automatic orientation determination**

---

## **Default Setup and Core Logic**

### **Camera Orientation**

* The **default setup** assumes the **camera is front-facing**, i.e., mounted inside the building and looking toward the entrance.
* The **curve** (the counting line) is drawn **across the entrance** — near the door threshold.
* When a person **walks toward the camera** (toward the entrance) and crosses the curve, they are labeled **IN**.
* When a person **walks away from the camera** and crosses the curve, they are labeled **OUT**.

> **Summary:**
>
> * **Toward camera → IN** (entering the building)
> * **Away from camera → OUT** (leaving the building)

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

* **Y-axis increases downward** (image coordinates).
* The **curve** spans the doorway area.
* People walking **upward (toward camera)** are **entering (IN)**.
* People walking **downward (away from camera)** are **exiting (OUT)**.

---

## **Camera Orientation Conventions**

| Camera Orientation | IN_direction (relative to curve) | Notes                                                                                    |
| ------------------ | -------------------------------- | ---------------------------------------------------------------------------------------- |
| Front-facing       | toward_cam                       | User sees people entering toward the camera.                                             |
| Back-facing        | away_from_cam                    | User sees people moving away from camera; still counted as IN if moving toward entrance. |
| Left-side          | left                             | IN = left side relative to the curve vector.                                             |
| Right-side         | right                            | IN = right side relative to the curve vector.                                            |
| Overhead           | left/right (relative to curve)   | User selects which side counts as IN; consistent screen-based labeling.                  |

> **Key Principle:** IN/OUT always refers to movement **toward the entrance**. Camera orientation determines which direction relative to the curve corresponds to IN.

---

## **Practical Example**

1. **Front-facing camera (default)**:

   * Person **outside** walks **toward camera** → IN.
   * Person **inside** walks **away from camera** → OUT.

2. **Back-facing camera**:

   * Person **outside** walks **away from camera toward entrance** → IN.
   * Person **inside** walks **toward camera (leaving entrance)** → OUT.

3. **Side-facing camera**:

   * Must define which side of the curve counts as IN (left or right).

4. **Overhead camera**:

   * User chooses IN side; labeling is consistent for all detected trajectories.

---

## **Counting Mechanism**

1. **Curve Definition**
   Defined in `config/curve_config.json` as a polyline across the entrance. Optionally, it can include `IN_direction` (front/back/left/right).

2. **Signed Distance Calculation**
   Each detected person’s position (e.g., bounding box center) is compared to the curve using
   `signed_distance_to_curve(pt, curve)` → yields **positive/negative distance**.

3. **Auto Orientation**
   `auto_orient_curve(curve, trajectory_sample)` analyzes a subset of points to determine which side of the curve corresponds to **IN** when `IN_direction` is not predefined.

4. **Labeling**
   Once orientation is determined:

   * **Positive oriented distance → IN**
   * **Negative oriented distance → OUT**

---

## **Key Modules**

| Module                           | Purpose                                                                      |
| -------------------------------- | ---------------------------------------------------------------------------- |
| `modules/curve_utils.py`         | Loads and manages curve configurations.                                      |
| `modules/tracker_logic.py`       | Core geometric logic and `signed_distance_to_curve`.                         |
| `modules/orientation.py`         | Contains `auto_orient_curve` for determining IN/OUT direction automatically. |
| `utils/viz_tools.py`             | Visualization and debugging utilities.                                       |
| `main.py`                        | Entry point for live or recorded video processing.                           |
| `test_trajectory_orientation.py` | Offline testing for trajectory-based IN/OUT labeling.                        |

---

## **Test Mode (Offline Validation)**

Run offline trajectory tests:

```bash
python test_trajectory_orientation.py
```

* Loads curve from `config/curve_config.json`
* Loads/defines trajectory points
* Applies `auto_orient_curve`
* Outputs: raw signed distance, oriented distance, and label (**IN/OUT**) for each point

---

## **Parked / To-Revisit Items**

1. **Gray-zone hysteresis**

   * Introduce a pixel buffer around the curve to prevent miscounts due to small fluctuations (“flapping”).
   * Disabled in trajectory tests for simplicity.

2. **Auto-orientation check**

   * Tested for trajectory CSVs.
   * May exhibit bias if initial movement samples are unbalanced (e.g., “morning bias”).

3. **Dynamic camera flip detection**

   * Detect camera rotation, flip, or mirror changes during operation.

4. **World-space mapping / real-world coordinates**

   * Future enhancement for multi-camera setups or moving cameras.

---

## **Next Steps**

* Validate IN/OUT labeling using recorded or live footage.
* Integrate gray-zone hysteresis and dynamic orientation updates.
* Add real-time visualization overlays to verify classification accuracy.

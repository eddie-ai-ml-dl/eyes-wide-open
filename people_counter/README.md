Absolutely! Here's a **clean, accurate, and coherent version of README.md**, reflecting all of our discussions and clarifications:

---

# Person Counting via Curve Crossing

## **Overview**

This application counts people moving across a predefined curve in a video feed. It supports **camera orientation-agnostic counting**, i.e., the system can handle cameras facing **inside, outside, or alongside** entrances.

It tracks people using a trajectory of points detected in the video, computes the **signed distance** of each point to the curve, and determines whether each person is moving **IN** (entering) or **OUT** (exiting).

---

## **Key Concepts**

1. **Trajectory Points**

   * Each person is represented as a sequence of 2D points `(x, y)` detected over time.

2. **Curve Definition**

   * The curve represents the boundary of interest (e.g., a doorway).
   * Stored as a list of 2D points in JSON format.

3. **Signed Distance to Curve**

   * Computed using `signed_distance_to_curve(point, curve)`.
   * Positive or negative values indicate which side of the curve the point is on.

4. **Auto-orientation of Curve**

   * `auto_orient_curve(curve_points)` determines the correct orientation so that distances are consistently signed.
   * Orientation is **relative**, not absolute: it standardizes calculations independent of curve drawing direction.

5. **Labeling IN / OUT**

   * The IN/OUT label depends on **camera placement** and the **direction of movement relative to the curve**.
   * A configurable function maps `(camera_placement, curve_orientation, signed_distance)` → IN/OUT.

---

## **Camera Placements & Rules**

1. **Inside-facing camera** (looking toward building interior)

   * Positive distance = moving toward building → IN
   * Negative distance = moving away from building → OUT

2. **Outside-facing camera** (looking toward exterior)

   * Positive distance = moving away from building → OUT
   * Negative distance = moving toward building → IN

3. **Alongside camera** (sideways view)

   * Define **which side is IN** (left or right).
   * Crossing from OUT side → IN side is counted as IN, otherwise OUT.

> The system is **camera orientation-agnostic** when combined with `auto_orient_curve()` and proper camera-side configuration.

---

## **Parked / To-Revisit Items**

1. **Gray-zone hysteresis**

   * Introduce a pixel buffer zone around the curve to prevent miscounts due to small fluctuations (“flapping”).
   * Currently disabled in trajectory tests for simplicity.

2. **Auto-orientation check**

   * Implemented and tested for trajectory CSVs.
   * Needs integration into the live video tracker pipeline.
   * ⚠️ Potential directional bias: If the calibration trajectories used for auto-orientation predominantly represent movement in one direction (e.g., morning entries), the computed orientation may reflect that bias rather than the true geometric curve orientation. Consider additional checks or multi-directional sampling to mitigate this.

3. **Dynamic camera flip detection**

   * Detect if the camera is physically reoriented (rotated, mirrored, flipped) during operation.
   * Could require dynamic orientation monitoring or short calibration steps.

4. **World-space mapping / real-world coordinates**

   * Optional enhancement to make counting independent of camera orientation and perspective.
   * Useful for multi-camera setups or if the camera is moved frequently.

---

## **Usage Example**

```python
from modules.curve_utils import load_curve_config
from modules.tracker_logic import signed_distance_to_curve
from modules.orientation import auto_orient_curve

# Load curve
curve = load_curve_config("config/curve_config.json")

# Determine curve orientation
orientation, diag = auto_orient_curve(curve)

# Iterate through person trajectory points
for pt in trajectory:
    dist = signed_distance_to_curve(pt, curve)
    oriented_dist = dist * orientation
    label = label_point(oriented_dist, camera_placement="outside")
    print(f"{pt} -> dist={dist:.2f}, oriented={oriented_dist:.2f}, label={label}")
```

---

## **Notes**

* **IN/OUT labeling is dynamic**: depends on curve orientation and camera placement.
* **Auto-orientation** ensures consistent distance sign conventions for any curve direction.
* **Trajectory tests** validate the system for known sequences before integration with live video.


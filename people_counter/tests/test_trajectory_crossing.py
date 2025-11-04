import numpy as np
import cv2
from modules.curve_manager import CurveManager

# -------------------------------
# Curve points (same as before)
# -------------------------------
curve_points = np.array([
    [461.0, 373.0],
    [474.0, 412.0],
    [508.0, 424.0],
    [547.0, 424.0],
    [591.0, 424.0],
    [635.0, 427.0],
    [661.0, 423.0],
    [705.0, 390.0],
    [704.0, 370.0]
], dtype=np.float32)

# -------------------------------
# Sample trajectory
# -------------------------------
trajectory = np.array([
    [392,438],[397,438],[401,437],[409,436],[421,436],[434,435],[446,434],[443,434],
    [454,434],[464,433],[473,430],[482,429],[502,426],[514,423],[520,420],[528,417],
    [536,416],[540,414],[545,413],[554,414],[564,414],[573,413],[580,413],[585,414],
    [589,414],[593,413],[597,413],[601,412],[605,409],[608,407],[614,408]
], dtype=np.float32)

# -------------------------------
# Initialize CurveManager
# -------------------------------
region_depth = 50.0
h, w = 480, 960
cm = CurveManager("config/test_curve_trajectory.json")
cm.curve_data = {
    "curve_points": curve_points.tolist(),
    "IN_direction": "toward_cam",
    "orientation": 1
}

# Build oriented inside region
inside_region = cm.build_inside_region(
    curve_points,
    in_direction="toward_cam",
    frame_shape=(h, w),
    region_depth=region_depth,
    orientation=1
)

# -------------------------------
# Check trajectory points against polygon
# -------------------------------
entered = False
entry_frame = None
for idx, pt in enumerate(trajectory):
    # pointPolygonTest >0 means inside, 0=on edge, <0=outside
    result = cv2.pointPolygonTest(inside_region.astype(np.int32), tuple(pt), measureDist=False)
    if result >= 0 and not entered:
        entered = True
        entry_frame = idx

print(f"Trajectory entered inside region: {entered}")
if entered:
    print(f"First entry at trajectory index {entry_frame}, coordinates {trajectory[entry_frame]}")

# -------------------------------
# Visualization
# -------------------------------
canvas = np.zeros((h, w, 3), dtype=np.uint8) + 30

# Draw inside region
cv2.polylines(canvas, [inside_region.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.fillPoly(canvas, [inside_region.astype(np.int32)], (0, 200, 0, 50))

# Draw curve
for i, p in enumerate(curve_points):
    cv2.circle(canvas, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
    if i > 0:
        cv2.line(canvas, (int(curve_points[i-1,0]), int(curve_points[i-1,1])),
                 (int(p[0]), int(p[1])), (0, 255, 255), 1)

# Draw trajectory
for i in range(1, len(trajectory)):
    cv2.line(canvas, (int(trajectory[i-1,0]), int(trajectory[i-1,1])),
             (int(trajectory[i,0]), int(trajectory[i,1])), (255,0,0), 2)
    cv2.circle(canvas, (int(trajectory[i,0]), int(trajectory[i,1])), 2, (255,0,0), -1)

cv2.putText(canvas, "Inside Region (green) + Trajectory (blue)", (10,20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

cv2.imshow("Trajectory Crossing Test", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

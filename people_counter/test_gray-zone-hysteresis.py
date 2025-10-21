import csv
import numpy as np
from modules.curve_utils import load_curve_config

# -----------------------------
# CONFIG
# -----------------------------
CURVE_CONFIG_PATH = "config/curve_config.json"
GRAY_THRESHOLD = 0  # Pixels near the curve to ignore (0 = disabled)

# -----------------------------
# Load curve
# -----------------------------
user_curve_points = load_curve_config(CURVE_CONFIG_PATH)
if user_curve_points is None:
    print("❌ Curve config not found!")
    exit()

curve = np.array(user_curve_points, np.float32)

# -----------------------------
# Load trajectory CSV
# -----------------------------
def load_trajectory(csv_path):
    coords = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            x, y = float(row[0]), float(row[1])
            coords.append((x, y))
    return coords

trajectory_file = "data/test_trajectory.csv"  # or "data/test_traj_out.csv"
trajectory = load_trajectory(trajectory_file)

# -----------------------------
# Signed distance to polyline
# -----------------------------
def signed_distance_to_curve(pt, curve):
    """
    Returns signed distance from point to nearest segment in polyline.
    Positive = below curve, Negative = above curve
    """
    x, y = pt
    min_dist = float("inf")
    sign = 1
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i + 1]
        dx, dy = x2 - x1, y2 - y1

        # Project point onto segment
        if dx == dy == 0:
            proj_x, proj_y = x1, y1
        else:
            t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy

        dist = np.sqrt((x - proj_x)**2 + (y - proj_y)**2)

        if dist < min_dist:
            min_dist = dist
            # Use cross product to determine sign
            vec_curve = np.array([dx, dy])
            vec_point = np.array([x - x1, y - y1])
            cross = vec_curve[0]*vec_point[1] - vec_curve[1]*vec_point[0]
            sign = -1 if cross > 0 else 1
    return min_dist * sign

# -----------------------------
# Simulation / Counting
# -----------------------------
track_state = {'last_sign': None, 'counted': False}
in_count = 0
out_count = 0

for anchor in trajectory:
    dist_signed = signed_distance_to_curve(anchor, curve)

    # Apply optional gray-zone
    if abs(dist_signed) < GRAY_THRESHOLD:
        current_sign = 0
    else:
        current_sign = np.sign(dist_signed)

    # Count crossing if last_sign exists and current_sign != 0
    if track_state['last_sign'] is not None and not track_state['counted'] and current_sign != 0:
        if track_state['last_sign'] > 0 and current_sign < 0:
            in_count += 1
            track_state['counted'] = True
        elif track_state['last_sign'] < 0 and current_sign > 0:
            out_count += 1
            track_state['counted'] = True

    # Update last_sign only if current_sign != 0
    if current_sign != 0:
        track_state['last_sign'] = current_sign

# -----------------------------
# Output
# -----------------------------
print(f"\nTrajectory result:")
print(f"IN count: {in_count}")
print(f"OUT count: {out_count}")

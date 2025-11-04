import numpy as np
import cv2
import os
from modules.curve_manager import CurveManager

# -------------------------------
# Input curve (provided)
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
# Test parameters
# -------------------------------
region_depth = 50.0
h, w = 480, 960
cfg_path = "config/test_curve_behavior.json"
os.makedirs(os.path.dirname(cfg_path), exist_ok=True)

cm = CurveManager(cfg_path)
cm.curve_data = {
    "curve_points": curve_points.tolist(),
    "IN_direction": "auto",
    "orientation": None,
    "orientation_diagnostics": {}
}

# --- 1) Neutral symmetric region ---
poly_sym = cm.build_inside_region(
    curve_points,
    in_direction="auto",
    frame_shape=(h, w),
    region_depth=region_depth,
    orientation=None
)
cm.save_inside_region(
    poly_sym,
    region_depth=region_depth,
    region_diag={
        "method": "normal_offset",
        "region_depth": region_depth,
        "num_points": int(len(poly_sym)),
        "status": "ok (symmetric)"
    }
)
print("\n--- Neutral symmetric region ---")
print(f"Polygon points: {len(poly_sym)}")
print(f"Bounding box: x∈[{poly_sym[:,0].min():.1f},{poly_sym[:,0].max():.1f}] "
      f"y∈[{poly_sym[:,1].min():.1f},{poly_sym[:,1].max():.1f}]")

# --- 2) Orientation = +1 (toward_cam) ---
poly_in = cm.build_inside_region(
    curve_points,
    in_direction="toward_cam",
    frame_shape=(h, w),
    region_depth=region_depth,
    orientation=1
)
cm.save_inside_region(
    poly_in,
    region_depth=region_depth,
    region_diag={
        "method": "normal_offset",
        "region_depth": region_depth,
        "num_points": int(len(poly_in)),
        "status": "ok (orientation=+1)"
    }
)
print("\n--- Orientation +1 (toward_cam) ---")
print(f"Polygon points: {len(poly_in)}")
print(f"Bounding box: x∈[{poly_in[:,0].min():.1f},{poly_in[:,0].max():.1f}] "
      f"y∈[{poly_in[:,1].min():.1f},{poly_in[:,1].max():.1f}]")

# --- 3) Orientation = -1 (away_from_cam) ---
poly_out = cm.build_inside_region(
    curve_points,
    in_direction="away_from_cam",
    frame_shape=(h, w),
    region_depth=region_depth,
    orientation=-1
)
cm.save_inside_region(
    poly_out,
    region_depth=region_depth,
    region_diag={
        "method": "normal_offset",
        "region_depth": region_depth,
        "num_points": int(len(poly_out)),
        "status": "ok (orientation=-1)"
    }
)
print("\n--- Orientation -1 (away_from_cam) ---")
print(f"Polygon points: {len(poly_out)}")
print(f"Bounding box: x∈[{poly_out[:,0].min():.1f},{poly_out[:,0].max():.1f}] "
      f"y∈[{poly_out[:,1].min():.1f},{poly_out[:,1].max():.1f}]")

# -------------------------------
# Visualization
# -------------------------------
canvas = np.zeros((h, w, 3), dtype=np.uint8) + 30

# Draw each region in different colors
vis_sym = cm.visualize_region(canvas, color=(0, 200, 0), alpha=0.3)
cv2.polylines(vis_sym, [poly_sym.astype(np.int32)], True, (0,255,0), 2, cv2.LINE_AA)

vis_in = cm.visualize_region(canvas, color=(0, 0, 255), alpha=0.3)
cv2.polylines(vis_in, [poly_in.astype(np.int32)], True, (0,0,255), 2, cv2.LINE_AA)

vis_out = cm.visualize_region(canvas, color=(255, 0, 0), alpha=0.3)
cv2.polylines(vis_out, [poly_out.astype(np.int32)], True, (255,0,0), 2, cv2.LINE_AA)

# Stack for comparison
stack = np.hstack([vis_sym, vis_in, vis_out])

# Draw original curve on top
for i, p in enumerate(curve_points):
    p_int = (int(p[0]), int(p[1]))
    cv2.circle(stack, p_int, 3, (0,255,255), -1)
    if i>0:
        cv2.line(stack, (int(curve_points[i-1,0]), int(curve_points[i-1,1])),
                 (int(p[0]), int(p[1])), (0,255,255), 1)

cv2.putText(stack, "Symmetric (auto)", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
cv2.putText(stack, "Orientation +1", (w+30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.putText(stack, "Orientation -1", (2*w+30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

cv2.imshow("Inside Region Tests", stack)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print one sample coordinate difference to verify offset distance
dist = np.linalg.norm(poly_in[0] - curve_points[0])
print(f"\nSample offset distance for orientation +1 first point: {dist:.2f} px (expected ~{region_depth})")

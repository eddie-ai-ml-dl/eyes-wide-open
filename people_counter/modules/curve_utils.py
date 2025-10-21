import cv2
import json
import numpy as np

# -----------------------------------------
# Curve utilities: setup, load, save, geometry
# -----------------------------------------

def save_curve_config(points, filepath):
    with open(filepath, 'w') as f:
        json.dump({'curve_points': points}, f, indent=4)
    print(f"✅ Curve configuration saved to {filepath}")

def load_curve_config(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)['curve_points']
    except Exception:
        return None

def interactive_curve_creator(frame, window_name="Curve Setup"):
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = frame.copy()
        cv2.putText(temp,
                    "Draw curve (L-click add, R-click undo, 's' save, ESC cancel)",
                    (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if len(points) > 1:
            cv2.polylines(temp, [np.array(points, np.int32)], False, (0,255,255), 2)
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s') and len(points) > 1:
            break
        if key == 27:
            return None
    cv2.destroyWindow(window_name)
    return points


# --- Geometry helpers ---
def point_distance_to_curve(pt, curve):
    x, y = pt
    distances = []
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i + 1]
        px = x2 - x1
        py = y2 - y1
        norm = px*px + py*py
        u = max(0, min(1, ((x - x1)*px + (y - y1)*py) / norm))
        dx = x1 + u*px - x
        dy = y1 + u*py - y
        distances.append(np.sqrt(dx*dx + dy*dy))
    return min(distances)

def _orient(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _on_segment(a,b,c):
    return min(a[0],c[0]) <= b[0] <= max(a[0],c[0]) and min(a[1],c[1]) <= b[1] <= max(a[1],c[1])

def segments_intersect(p1,p2,q1,q2):
    o1 = _orient(p1,p2,q1)
    o2 = _orient(p1,p2,q2)
    o3 = _orient(q1,q2,p1)
    o4 = _orient(q1,q2,p2)
    if o1*o2 < 0 and o3*o4 < 0:
        return True
    if abs(o1)<1e-9 and _on_segment(p1,q1,p2): return True
    if abs(o2)<1e-9 and _on_segment(p1,q2,p2): return True
    if abs(o3)<1e-9 and _on_segment(q1,p1,q2): return True
    if abs(o4)<1e-9 and _on_segment(q1,p2,q2): return True
    return False

def path_crosses_curve(prev_anchor, cur_anchor, curve_poly):
    for i in range(len(curve_poly)-1):
        if segments_intersect(prev_anchor, cur_anchor, tuple(curve_poly[i]), tuple(curve_poly[i+1])):
            return True
    return False

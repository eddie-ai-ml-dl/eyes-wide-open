import numpy as np

# -----------------------------
# Config
# -----------------------------
GRAY_ZONE_WIDTH = 25

# -----------------------------
# Signed distance to curve
# -----------------------------
def signed_distance_to_curve(pt, curve):
    x, y = pt
    min_dist = float("inf")
    sign = 1
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i+1]

        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:
            proj_x, proj_y = x1, y1
        else:
            t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)))
            proj_x = x1 + t*dx
            proj_y = y1 + t*dy

        dist = np.sqrt((x - proj_x)**2 + (y - proj_y)**2)
        if dist < min_dist:
            min_dist = dist
            # Cross product to determine side
            vec_curve = np.array([dx, dy])
            vec_point = np.array([x - x1, y - y1])
            cross = vec_curve[0]*vec_point[1] - vec_curve[1]*vec_point[0]
            sign = -1 if cross > 0 else 1
    return min_dist * sign

# -----------------------------
# Classify region
# -----------------------------
def classify_region(anchor, curve):
    dist_signed = signed_distance_to_curve(anchor, curve)
    return np.sign(dist_signed)

# -----------------------------
# Update track state
# -----------------------------
def update_track_state(track_id, anchor, track_states, curve, inside_region, in_count, out_count):
    if track_id not in track_states:
        track_states[track_id] = {
            'last_sign': None,
            'counted': False,
            'path': []
        }

    state = track_states[track_id]
    state['path'].append(anchor)
    current_sign = classify_region(anchor, curve)

    if state['last_sign'] is not None and not state['counted']:
        # Sign change = crossing
        if state['last_sign'] > 0 and current_sign < 0:
            in_count += 1
            state['counted'] = True
        elif state['last_sign'] < 0 and current_sign > 0:
            out_count += 1
            state['counted'] = True

    state['last_sign'] = current_sign
    return in_count, out_count

import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from modules.curve_manager import CurveManager
from modules.tracker_logic import update_track_state, signed_distance_to_curve
from utils.viz_tools import plot_trajectory
import json

# -------------------------
# CONFIG
# -------------------------
CURVE_CONFIG_PATH = "config/curve_config.json"
FRAME_SKIP = 1  # process every nth frame

# -------------------------
# Load model & video
# -------------------------
model = YOLO("models/yolo12n.pt")
cap = cv2.VideoCapture("../data/videos/People Entering And Exiting Mall Stock Footage.mp4")

ret, first_frame = cap.read()
if not ret:
    print("❌ Cannot read video")
    exit()

first_frame = cv2.resize(first_frame, (1020, 600))
h, w = first_frame.shape[:2]

# -------------------------
# Load or create curve
# -------------------------
cm = CurveManager(CURVE_CONFIG_PATH)
curve_data = cm.load_curve_config()
if not curve_data:
    # Config missing, create interactively
    curve_data = cm.create_curve(first_frame)
    if not curve_data:
        print("Curve creation cancelled.")
        exit()

user_curve_np = np.array(curve_data["curve_points"], dtype=np.float32)

# -------------------------
# State tracking
# -------------------------
TRACK_ID = 128
trajectory_data = {"id": TRACK_ID, "path": []}
track_states = {}
in_count = out_count = 0
frame_idx = 0
orientation_determined = False
sample_anchors = []
inside_region = None
resolved_in_direction = None
orientation = None

# -------------------------
# Video processing loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (w, h))
    cv2.polylines(frame, [user_curve_np.astype(int)], False, (0, 255, 255), 2)

    results = model.track(frame, persist=True, classes=[0], verbose=False, tracker="botsort.yaml")

    anchors_in_frame = []
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            anchor = (float((x1 + x2) / 2), float(y2))
            anchors_in_frame.append(anchor)

    # -------------------------
    # Collect sample points for auto-orientation if needed
    # -------------------------
    if not orientation_determined:
        sample_anchors.extend(anchors_in_frame)
        if len(sample_anchors) >= 10:
            orientation, diag, resolved_in_direction = cm.determine_orientation(sample_anchors)

            # Round diagnostics numbers to 2 decimal places
            # if diag:
            #     def round_recursive(d):
            #         if isinstance(d, dict):
            #             return {k: round_recursive(v) for k, v in d.items()}
            #         elif isinstance(d, list):
            #             return [round_recursive(v) for v in d]
            #         elif isinstance(d, float):
            #             return round(d, 2)
            #         else:
            #             return d
            #     diag_rounded = round_recursive(diag)
            #     curve_data["orientation_diagnostics"] = diag_rounded
            #     cm.save_curve_config(curve_data)

            # Build inside_region now that direction is resolved
            inside_region = CurveManager.build_inside_region(user_curve_np, resolved_in_direction, (h, w))

            orientation_determined = True
            print(f"Orientation determined: {orientation}, IN_direction: {resolved_in_direction}")
    else:
        # Build inside_region once if not yet built (for subsequent frames)
        if inside_region is None:
            inside_region = CurveManager.build_inside_region(user_curve_np, resolved_in_direction, (h, w))

        # -------------------------
        # Update track states and counting
        # -------------------------
        for track_id, box, anchor in zip(ids, boxes, anchors_in_frame):
            x1, y1, x2, y2 = box
            dist_signed = signed_distance_to_curve(anchor, user_curve_np) * orientation
            current_region = "INSIDE" if dist_signed < 0 else "OUTSIDE"

            if track_id.item() == TRACK_ID:
                cx, cy = int((x1 + x2) / 2), int(y2)
                trajectory_data["path"].append((cx, cy))

            in_count, out_count = update_track_state(
                track_id, anchor, track_states, user_curve_np, inside_region, in_count, out_count
            )

            # Visuals
            color = (255, 255, 0)
            if track_states[track_id]['counted']:
                color = (0, 255, 0)
            if track_id.item() == TRACK_ID:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(frame, f"ID:{track_id}", (x1, y1 - 10), 1, 1)

    cvzone.putTextRect(frame, f"IN: {in_count}", (40, 60), 2, 2, colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f"OUT: {out_count}", (40, 110), 2, 2, colorR=(0, 0, 255))

    cv2.imshow("Person Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nFinal IN Count: {in_count}\nFinal OUT Count: {out_count}")
print(f"Trajectory path length: {len(trajectory_data['path'])}")
plot_trajectory(trajectory_data)

import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from modules.curve_utils import load_curve_config, save_curve_config, interactive_curve_creator
from modules.tracker_logic import update_track_state, signed_distance_to_curve
from modules.orientation import auto_orient_curve
from utils.viz_tools import plot_trajectory

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
out_persons = []

# -------------------------
# Load / create curve
# -------------------------
user_curve_points = load_curve_config(CURVE_CONFIG_PATH)
if not user_curve_points:
    user_curve_points = interactive_curve_creator(first_frame)
    if not user_curve_points:
        print("Curve setup cancelled."); exit()
    save_curve_config(user_curve_points, CURVE_CONFIG_PATH)

user_curve_np = np.array(user_curve_points, np.float32)
h, w = 600, 1020

# Build INSIDE polygon (bottom side)
inside_corners = np.array([
    [user_curve_np[user_curve_np[:, 0].argmax()][0], h],
    [user_curve_np[user_curve_np[:, 0].argmin()][0], h]
])
INSIDE_REGION = np.concatenate((user_curve_np, inside_corners), axis=0)

# -------------------------
# State tracking
# -------------------------
TRACK_ID = 128
trajectory_data = {"id": TRACK_ID,
                   "path": [],
                  }
track_states = {}
in_count = out_count = 0
frame_idx = 0
orientation_determined = False
sample_anchors = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (w, h))
    cv2.polylines(frame, [user_curve_np.astype(int)], False, (0,255,255), 2)

    results = model.track(frame, persist=True, classes=[0], verbose=False, tracker="botsort.yaml") #"config/custom_botsort.yaml") #"botsort.yaml") #"bytetrack.yaml")

    # -------------------------
    # Collect sample points for auto-orientation
    # -------------------------
    anchors_in_frame = []
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            anchor = (float((x1 + x2)/2), float(y2))
            anchors_in_frame.append(anchor)

    if not orientation_determined:
        sample_anchors.extend(anchors_in_frame)
        if len(sample_anchors) >= 10:  # collect 10 anchors then compute orientation
            orientation, diag=auto_orient_curve(user_curve_np, sample_anchors, eps=3.0, min_crossings=1)
            print(f"Auto-detected orientation: {orientation}, diag={diag}")
            orientation_determined = True
            print(f"Auto-detected curve orientation: {orientation:+d}")
    else:
        # -------------------------
        # Update track states and counting
        # -------------------------
        for track_id, box, anchor in zip(ids, boxes, anchors_in_frame):
            x1, y1, x2, y2 = box
            # Multiply by orientation
            dist_signed = signed_distance_to_curve(anchor, user_curve_np) * orientation
            current_region = 'INSIDE' if dist_signed < 0 else 'OUTSIDE'

            if track_id.item() == TRACK_ID:
                cx, cy=int((x1+x2)/2), int(y2)
                trajectory_data["path"].append((cx, cy))

            x = out_count
            in_count, out_count = update_track_state(
                track_id, anchor, track_states, user_curve_np, INSIDE_REGION, in_count, out_count
            )
            if out_count > x:
                out_persons.append(track_id.item())
            # Visuals
            color = (255,255,0)
            if track_states[track_id]['counted']:
                color = (0,255,0)
            if track_id.item() == TRACK_ID:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(frame, f"ID:{track_id}", (x1, y1-10), 1, 1)

    cvzone.putTextRect(frame, f"IN: {in_count}", (40,60), 2, 2, colorR=(0,128,0))
    cvzone.putTextRect(frame, f"OUT: {out_count}", (40,110), 2, 2, colorR=(0,0,255))

    cv2.imshow("Person Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal IN Count: {in_count}\nFinal OUT Count: {out_count}")
print(f"OUT IDs: {out_persons}")
print(trajectory_data["path"])
plot_trajectory(trajectory_data["path"])

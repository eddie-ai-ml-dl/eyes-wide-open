import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from modules.curve_utils import (
    load_curve_config, save_curve_config, interactive_curve_creator
)
from modules.tracker_logic import update_track_state

# -------------------------
# Main entry point
# -------------------------
CURVE_CONFIG_PATH = "config/curve_config.json"

# Load model & video
model = YOLO("yolo12n.pt")
cap = cv2.VideoCapture("../data/videos/People Entering And Exiting Mall Stock Footage.mp4")

ret, first_frame = cap.read()
if not ret:
    print("❌ Cannot read video")
    exit()

first_frame = cv2.resize(first_frame, (1020, 600))
user_curve_points = load_curve_config(CURVE_CONFIG_PATH)
if not user_curve_points:
    user_curve_points = interactive_curve_creator(first_frame)
    if not user_curve_points:
        print("Curve setup cancelled."); exit()
    save_curve_config(user_curve_points, CURVE_CONFIG_PATH)

user_curve_np = np.array(user_curve_points, np.int32)
h, w = 600, 1020

# Build INSIDE polygon (bottom side)
inside_corners = np.array([
    [user_curve_np[user_curve_np[:, 0].argmax()][0], h],
    [user_curve_np[user_curve_np[:, 0].argmin()][0], h]
])
INSIDE_REGION = np.concatenate((user_curve_np, inside_corners), axis=0)

# --- State tracking ---
track_states = {}
in_count = out_count = 0

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % 2 != 0:
        continue

    frame = cv2.resize(frame, (w, h))
    cv2.polylines(frame, [user_curve_np], False, (0,255,255), 2)

    results = model.track(frame, persist=True, classes=[0], verbose=False, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            anchor = (float((x1 + x2) / 2), float(y2))

            in_count, out_count = update_track_state(
                track_id, anchor, track_states, user_curve_np, INSIDE_REGION, in_count, out_count
            )

            color = (255,255,0)
            if track_states[track_id]['counted']:
                color = (0,255,0)
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

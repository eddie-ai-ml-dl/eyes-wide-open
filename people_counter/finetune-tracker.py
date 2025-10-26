import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from copy import deepcopy
import itertools
import yaml
import tempfile

from modules.curve_utils import load_curve_config, interactive_curve_creator, save_curve_config
from modules.tracker_logic import update_track_state

# -----------------------------
# CONFIG
# -----------------------------
CURVE_CONFIG_PATH = "config/curve_config.json"
VIDEO_PATH = "../data/videos/People Entering And Exiting Mall Stock Footage.mp4"

# Base tracker config template
BASE_TRACKER_CONFIG_PATH = "config/custom_botsort.yaml"

# Sweep ranges
TRACK_BUFFER_VALUES = [30, 50, 60, 80, 100]
NEW_TRACK_THRESH_VALUES = [0.15, 0.2, 0.25, 0.3, 0.35]

# -----------------------------
# Load curve
# -----------------------------
user_curve_points = load_curve_config(CURVE_CONFIG_PATH)
if not user_curve_points:
    # Interactive setup if missing
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Cannot read video")
        exit()
    first_frame = cv2.resize(first_frame, (1020, 600))
    user_curve_points = interactive_curve_creator(first_frame)
    if user_curve_points:
        save_curve_config(user_curve_points, CURVE_CONFIG_PATH)
    else:
        print("Curve setup cancelled."); exit()

user_curve_np = np.array(user_curve_points, np.int32)
h, w = 600, 1020
inside_corners = np.array([
    [user_curve_np[user_curve_np[:,0].argmax()][0], h],
    [user_curve_np[user_curve_np[:,0].argmin()][0], h]
])
INSIDE_REGION = np.concatenate((user_curve_np, inside_corners), axis=0)

# -----------------------------
# Load base tracker config
# -----------------------------
with open(BASE_TRACKER_CONFIG_PATH) as f:
    base_tracker_config = yaml.safe_load(f)

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolo12n.pt")

# -----------------------------
# Run parameter sweep
# -----------------------------
results_summary = []

for track_buffer, new_track_thresh in itertools.product(TRACK_BUFFER_VALUES, NEW_TRACK_THRESH_VALUES):
    # Deepcopy config and update parameters
    tracker_config = deepcopy(base_tracker_config)
    tracker_config['track_buffer'] = track_buffer
    tracker_config['new_track_thresh'] = new_track_thresh

    # Write temporary YAML for this iteration
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp:
        yaml.safe_dump(tracker_config, tmp)
        tmp.flush()

        # Reset video and tracking state
        cap = cv2.VideoCapture(VIDEO_PATH)
        track_states = {}
        in_count = 0
        out_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 4 != 0:
                continue  # skip every other frame

            frame = cv2.resize(frame, (w, h))
            results = model.track(frame, persist=True, classes=[0], verbose=False, tracker=tmp.name)

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for track_id, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box
                    anchor = (float((x1 + x2)/2), float(y2))
                    in_count, out_count = update_track_state(
                        track_id, anchor, track_states, user_curve_np, INSIDE_REGION, in_count, out_count
                    )

        cap.release()

    results_summary.append({
        'track_buffer': track_buffer,
        'new_track_thresh': new_track_thresh,
        'IN': in_count,
        'OUT': out_count
    })
    print(f"Tested track_buffer={track_buffer}, new_track_thresh={new_track_thresh} => IN={in_count}, OUT={out_count}")

# -----------------------------
# Summary table
# -----------------------------
print("\n=== Parameter Sweep Summary ===")
print(f"{'TrackBuffer':>12} {'NewTrackThresh':>15} {'IN':>5} {'OUT':>5}")
for res in results_summary:
    print(f"{res['track_buffer']:>12} {res['new_track_thresh']:>15} {res['IN']:>5} {res['OUT']:>5}")

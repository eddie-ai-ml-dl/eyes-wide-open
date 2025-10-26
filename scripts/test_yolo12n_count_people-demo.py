# Import required libraries
import cv2
from ultralytics import YOLO
import cvzone

# Load YOLO V12 model
model = YOLO('yolo12n.pt')
names=model.names
# Define vertical line's X position
line_x = 500
line_y = 200

# Track previous center positions
track_history = {}

# IN/OUT counters
in_count = 0
out_count = 0

in_ids, out_ids = [], []

# Open video file or webcam
cap = cv2.VideoCapture("../data/videos/People Entering And Exiting Mall Stock Footage.mp4") # Use 0 for webcam

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [[{x}, {y}]]")

# Create a named OpenCV window and set the mouse callback
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)
frame_count=0
while True:
    # Read video frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 600))
    line_x=frame.shape[1]//2
    line_y=frame.shape[0]//2 + 60

    # Detect and track persons (class 0)
    results = model.track(frame, persist=True, classes=[0], verbose=False)
    # annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        # print("Detected: ", len(ids), " | ", ids)
        for track_id, box, class_id in zip(ids,boxes,class_ids):
            x1,y1,x2,y2=box
            name=names[class_id]
            cx=int((x1+x2)/2)
            cy= y1 #(y1+y2)//2 #y2-2
            rectangle_rgb = (255,0,0)
            if track_id == 7:
                print(f"{cx},{cy}")
            #     rectangle_rgb = (0,0,255)
            #     # print(f"Tracking ===>>> {track_id} | {line_y} | box: {box} | cy: {cy}")
            #     cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            #     cv2.rectangle(frame,(x1,y1),(x2,y2),rectangle_rgb,2)
            #     cvzone.putTextRect(frame,f'{track_id}',(x1,y1),1,1)

            if track_id in track_history:
                prev_cx, prev_cy = track_history[track_id]
                if (prev_cy < line_y < cy):
                    in_count += 1
                    in_ids.append((track_id, frame_count))
                    cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(frame,f'{track_id}',(x1,y1),1,1)
                if (prev_cy>line_y>cy):
                    out_count+=1
                    out_ids.append(track_id)
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
            track_history[track_id] = (cx,cy)

    # Display counts using cvzone's putTextRect
    cvzone.putTextRect(frame, f'IN: {in_count}', (40,60), scale=2, thickness=2, colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (40,100), scale=2, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))
    cv2.line(frame, (line_x,0),(line_x,frame.shape[0]),(255,255,255),2)
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255,255,255),2)

    # Show the frame
    cv2.imshow("RGB", frame)
    # print(track_history)
    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
import numpy as np
print(f"IN IDs: {np.array(in_ids).tolist()}")
print(f"OUT IDs: {np.array(out_ids).tolist()}")
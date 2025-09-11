import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo12n.pt")

# Open the video file
video_path = "../data/videos/People Entering And Exiting Mall Stock Footage.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        frame=cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        print("Detected:", len(results[0].boxes.id.cpu().numpy().astype(int)))
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
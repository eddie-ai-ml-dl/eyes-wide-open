# yolov12_testing_project/scripts/test_yolov12_videos.py

from ultralytics import YOLO
import os

# --- Configuration ---
# Path to the directory containing your input test videos
VIDEO_DIR=os.path.join('..', 'data', 'videos')
# Path to the directory where Ultralytics will save annotated video files
INFERENCE_OUTPUT_DIR=os.path.join('..', 'runs', 'detect')

# Name of the pre-trained YOLOv12 model to use
MODEL_NAME='yolo12n.pt'

# Minimum confidence score for a detection to be considered valid
CONFIDENCE_THRESHOLD=0.5


def test_yolov12_video_detection():
    """
    Loads a YOLOv12 model, runs inference on videos in VIDEO_DIR,
    detects 'person' objects, and saves the annotated videos.
    Prints a summary of person detections for each video.
    """
    print("--- Starting YOLOv12 Video Detection Test ---")
    print(f"Model: {MODEL_NAME}, Confidence Threshold: {CONFIDENCE_THRESHOLD}")

    # --- Ensure output directory exists ---
    # Ultralytics will create a subfolder like 'yolov12_video_detection' inside this.
    try:
        os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)
        print(f"Ensured '{INFERENCE_OUTPUT_DIR}' exists for annotated video output.")
    except OSError as e:
        print(f"Error creating output directory: {e}. Exiting.")
        return

    # --- Load YOLOv12 Model ---
    print(f"Attempting to load YOLOv12 model: {MODEL_NAME}")
    try:
        model=YOLO(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{MODEL_NAME}': {e}")
        print("Please ensure you have an active internet connection for first-time download,")
        print("or that the model file exists if loading locally. Exiting.")
        return

    # --- Gather Video Files ---
    video_files=[os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
                 if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]

    if not video_files:
        print(f"No video files found in '{VIDEO_DIR}'. Please add some videos to test.")
        return

    print(f"Found {len(video_files)} videos to process in '{VIDEO_DIR}'.")

    total_videos_processed=0

    try:
        # --- Run Inference on each video ---
        for video_path in video_files:
            video_filename=os.path.basename(video_path)
            print(f"\n--- Running detection for video: {video_filename} ---")

            # The 'predict' method with save=True will save the annotated video directly.
            # 'project' and 'name' control the output folder structure.
            # 'show=False' prevents a display window from popping up for each video frame.
            # 'stream=True' is often used for real-time applications, but for saving a full video,
            # Ultralytics handles the frame-by-frame processing and saving automatically.

            # The 'results' object will be an iterator yielding frame-level results.
            # We can iterate through it to get frame-by-frame detections.

            person_detections_per_frame=[]  # To store count of persons per frame

            # Note: For video, model.predict returns an iterator.
            # The annotated video file will be saved by Ultralytics automatically due to `save=True`.
            for frame_idx, result in enumerate(model.predict(
                    source=video_path,
                    save=True,
                    project=INFERENCE_OUTPUT_DIR,
                    name='yolov12_video_detection',  # Output folder: runs/detect/yolov12_video_detection/
                    conf=CONFIDENCE_THRESHOLD,
                    show=False,
                    verbose=False)):  # Set to True for more detailed console output during inference per frame

                person_count_in_frame=0
                if result.boxes:
                    for box in result.boxes:
                        class_id=int(box.cls[0])
                        label=model.names[class_id]
                        if label=='person':
                            person_count_in_frame+=1
                person_detections_per_frame.append(person_count_in_frame)

                # Print progress every N frames
                if (frame_idx+1)%100==0:
                    print(f"  Processed {frame_idx+1} frames for {video_filename}...")

            total_frames=len(person_detections_per_frame)
            if total_frames>0:
                avg_persons_per_frame=sum(person_detections_per_frame)/total_frames
                max_persons_in_frame=max(person_detections_per_frame)
                print(f"  Video '{video_filename}' processed {total_frames} frames.")
                print(f"  Average persons detected per frame: {avg_persons_per_frame:.2f}")
                print(f"  Maximum persons detected in a single frame: {max_persons_in_frame}")
            else:
                print(f"  No frames processed or detections found for '{video_filename}'.")

            total_videos_processed+=1

    except Exception as e:
        print(f"\nAn unexpected error occurred during video inference: {e}")
        print("Hint: Ensure OpenCV (cv2) is properly installed, as Ultralytics often uses it for video handling.")
        print("      Try: pip install opencv-python")

    print(f"\n--- Video Detection Test Complete ---")
    print(f"Total videos processed: {total_videos_processed}")
    output_path=os.path.join(INFERENCE_OUTPUT_DIR, 'yolov12_video_detection')
    print(f"Annotated videos are saved in: '{output_path}'")
    print("You can find the processed videos (e.g., 'your_video_name.mp4') inside this folder.")


if __name__=="__main__":
    test_yolov12_video_detection()
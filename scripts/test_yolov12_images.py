from ultralytics import YOLO
import os

# Define the path to your test images
IMAGE_DIR=os.path.join('..', 'data', 'images')
OUTPUT_DIR=os.path.join('..', 'runs', 'detect')  # Ultralytics typically creates subfolders here

# Path to the YOLOv12 model weights
# 'yolov12.pt' for detection, 'yolov12-seg.pt' for segmentation, etc.
MODEL_NAME='yolo12n.pt'


def test_yolov12_image_detection():
    print(f"Loading YOLOv12 model: {MODEL_NAME}")
    try:
        # Load a pretrained YOLOv12 model
        model=YOLO(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an active internet connection or download the weights manually if needed.")
        return

    # Get a list of all image files in the directory
    image_files=[os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in {IMAGE_DIR}. Please add some images to test.")
        return

    print(f"Found {len(image_files)} images to test in {IMAGE_DIR}.")
    print("Running inference...")

    try:
        # Run inference on the images
        # The 'save=True' argument will save the annotated images to the 'runs/detect/predict' directory
        results=model.predict(source=image_files, save=True, project=OUTPUT_DIR, name='yolov12_image_test')

        print("\nInference complete!")
        output_path=os.path.join(OUTPUT_DIR, 'yolov12_image_test')
        print(f"Annotated images saved to: {output_path}")

        # Process and capture detection details
        all_detections=[]  # To store all detections across images

        for i, result in enumerate(results):
            image_filename=os.path.basename(image_files[i])
            print(f"\n--- Detections for: {image_filename} ---")

            # This list will store detections for the current image
            image_detections=[]

            if result.boxes:  # If object detection results are available
                # Iterate through each detected bounding box
                for box in result.boxes:
                    class_id=int(box.cls[0])  # Get the class ID (e.g., 0 for 'person', 1 for 'bicycle')
                    label=model.names[class_id]  # Map class ID to a human-readable label (e.g., 'person')
                    confidence=float(box.conf[0])  # Get the confidence score

                    detection_info={
                        "class_name": label,
                        "confidence": confidence,
                        "bbox_xyxy": box.xyxy[0].tolist(),  # Bounding box in [x1, y1, x2, y2] format
                        "image_path": image_files[i]  # Useful for tracing back
                    }
                    image_detections.append(detection_info)
                    all_detections.append(detection_info)

                    print(f"  - {label} (Confidence: {confidence:.2f}) [BBox: {box.xyxy[0].tolist()}]")
            else:
                print("  No objects detected.")

            # You can also store image_detections if you want results per image separately
            # For this example, we're just printing and aggregating to all_detections

        print("\n--- Summary of All Detections Across All Images ---")
        if all_detections:
            for detection in all_detections:
                print(
                    f"Image: {os.path.basename(detection['image_path'])}, Object: {detection['class_name']} (Conf: {detection['confidence']:.2f})")
        else:
            print("No objects were detected in any of the images.")

        # Optionally, save all_detections to a JSON file for later analysis
        # import json
        # with open(os.path.join(output_path, 'detections_summary.json'), 'w') as f:
        #     json.dump(all_detections, f, indent=4)
        # print(f"\nDetailed detection summary saved to: {os.path.join(output_path, 'detections_summary.json')}")

    except Exception as e:
        print(f"An error occurred during inference: {e}")


if __name__=="__main__":
    test_yolov12_image_detection()
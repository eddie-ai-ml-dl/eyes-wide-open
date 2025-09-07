from ultralytics import YOLO
import os
from PIL import Image  # Pillow library for image manipulation

# --- Configuration ---
# Path to the directory containing your input test images
IMAGE_DIR=os.path.join('..', 'data', 'images')
# Path to the directory where cropped 'person' images will be saved
CROPPED_OUTPUT_DIR=os.path.join('..', 'data', 'cropped_persons')
# Path to the directory where Ultralytics will save annotated (boxes drawn on) images
INFERENCE_OUTPUT_DIR=os.path.join('..', 'runs', 'detect')

# Name of the pre-trained YOLOv12 model to use
# 'yolov12n.pt' for detection, 'yolov12n-seg.pt' for segmentation, etc.
MODEL_NAME='yolo12n.pt'

# Minimum confidence score for a detection to be considered valid and processed
# Detections below this threshold will be ignored.
CONFIDENCE_THRESHOLD=0.5


def extract_and_store_persons():
    """
    Loads a YOLOv12 model, runs inference on images in IMAGE_DIR,
    detects 'person' objects, crops them from the original images,
    and saves these cropped images to CROPPED_OUTPUT_DIR.
    It also saves annotated images to INFERENCE_OUTPUT_DIR.
    """
    print("--- Starting Person Extraction Process ---")

    # --- Ensure output directories exist ---
    try:
        os.makedirs(CROPPED_OUTPUT_DIR, exist_ok=True)
        os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)
        print(f"Ensured '{CROPPED_OUTPUT_DIR}' and '{INFERENCE_OUTPUT_DIR}' exist.")
    except OSError as e:
        print(f"Error creating output directories: {e}")
        return

    # --- Load YOLOv12 Model ---
    print(f"Attempting to load YOLOv12 model: {MODEL_NAME}")
    try:
        model=YOLO(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{MODEL_NAME}': {e}")
        print("Please ensure you have an active internet connection for first-time download,")
        print("or that the model file exists if loading locally.")
        return

    # --- Gather Image Files ---
    image_files=[os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in '{IMAGE_DIR}'. Please add some images to test.")
        return

    print(f"Found {len(image_files)} images to process in '{IMAGE_DIR}'.")
    print(f"Running inference with confidence threshold: {CONFIDENCE_THRESHOLD}")

    total_persons_extracted=0

    try:
        # --- Run Inference ---
        # 'save=True' will save the annotated images to a subfolder within INFERENCE_OUTPUT_DIR
        # 'project' and 'name' control the output folder structure for annotated images
        results=model.predict(source=image_files,
                              save=True,
                              project=INFERENCE_OUTPUT_DIR,
                              name='yolov12_image_person_extraction',
                              # This creates 'runs/detect/yolov12_image_person_extraction'
                              conf=CONFIDENCE_THRESHOLD,  # Apply confidence threshold
                              verbose=False)  # Set to True for more detailed console output during inference

        # --- Process Each Image's Results ---
        for i, result in enumerate(results):
            image_path=image_files[i]
            image_filename=os.path.basename(image_path)
            print(f"\n--- Processing detections for: {image_filename} ---")

            # Load the original image using PIL for cropping.
            # We explicitly open the original image here, not the annotated one.
            try:
                original_image=Image.open(image_path).convert("RGB")
            except Exception as img_e:
                print(f"  Error loading original image '{image_filename}': {img_e}. Skipping.")
                continue

            person_count_in_image=0

            if result.boxes:  # Check if any objects were detected in this image
                for j, box in enumerate(result.boxes):
                    class_id=int(box.cls[0])  # Get the class ID (e.g., 0 for 'person')
                    label=model.names[class_id]  # Map class ID to human-readable label (e.g., 'person')
                    confidence=float(box.conf[0])  # Get the confidence score

                    # Print detected object details (even if not 'person' or below threshold)
                    print(f"  - Detected: {label} (Confidence: {confidence:.2f}) [BBox: {box.xyxy[0].tolist()}]")

                    # --- Extract and Save 'person' Crops ---
                    if label=='person':
                        x1, y1, x2, y2=map(int, box.xyxy[0].tolist())  # Convert bbox to integer coordinates

                        # Ensure coordinates are within image bounds (important for robustness)
                        x1=max(0, x1)
                        y1=max(0, y1)
                        x2=min(original_image.width, x2)
                        y2=min(original_image.height, y2)

                        # Validate bounding box dimensions before cropping
                        if x2<=x1 or y2<=y1:
                            print(
                                f"    Warning: Invalid bounding box ({x1},{y1},{x2},{y2}) for person {j} in {image_filename}, skipping crop.")
                            continue

                        # Perform the image crop
                        cropped_person_image=original_image.crop((x1, y1, x2, y2))

                        # Create a unique filename for the cropped person
                        # Format: originalfilename_person_idx_conf_score.jpg
                        base_name=os.path.splitext(image_filename)[0]
                        cropped_filename=f"{base_name}_person_{j}_conf{confidence:.2f}.jpg"
                        cropped_filepath=os.path.join(CROPPED_OUTPUT_DIR, cropped_filename)

                        try:
                            cropped_person_image.save(cropped_filepath)
                            print(f"    -> Saved cropped person to: {cropped_filepath}")
                            person_count_in_image+=1
                            total_persons_extracted+=1
                        except Exception as save_e:
                            print(f"    Error saving cropped image '{cropped_filename}': {save_e}")
            else:
                print("  No objects detected in this image.")

            if person_count_in_image==0:
                print(f"  No 'person' objects (above confidence {CONFIDENCE_THRESHOLD}) detected in {image_filename}.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during inference or extraction: {e}")
        # Hint for common error if Pillow is missing
        if "No module named 'PIL'" in str(e):
            print("Hint: The 'Pillow' library is required for image manipulation. Install it with: pip install Pillow")

    print(f"\n--- Extraction Process Complete ---")
    print(f"Total 'person' images extracted: {total_persons_extracted}")
    print(f"Cropped images can be found in: '{CROPPED_OUTPUT_DIR}'")
    print(
        f"Annotated input images (with detections drawn) are saved in: '{os.path.join(INFERENCE_OUTPUT_DIR, 'yolov12_image_person_extraction')}'")


if __name__=="__main__":
    extract_and_store_persons()
# yolov12_testing_project/scripts/test_yolov12_img_cropping.py

from ultralytics import YOLO
import os
from PIL import Image  # Still needed here to open the original image
import datetime  # Still needed here to generate the timestamp for a new entry

# --- Import utility functions ---
from src.utils.image_processing import pad_to_square_and_resize
from src.utils.metadata_manager import save_metadata

# --- Configuration ---
IMAGE_DIR=os.path.join('..', 'data', 'images', 'phase1')
CROPPED_OUTPUT_DIR=os.path.join('..', 'data', 'cropped_persons')
INFERENCE_OUTPUT_DIR=os.path.join('..', 'runs', 'detect')

MODEL_NAME='yolo12n.pt'
CONFIDENCE_THRESHOLD=0.5

# New Parameters for Cropping Best Practices
CROP_PADDING_RATIO=0.10
TARGET_CLASSIFIER_SIZE=224
MIN_CROP_DIMENSION_PX=50

# Metadata Configuration
METADATA_FILENAME='cropped_persons_metadata.json'


def extract_and_store_persons_for_classification():
    """
    Loads a YOLOv12 model, runs inference on images in IMAGE_DIR,
    detects 'person' objects, applies padding, filters by size,
    pads to a square aspect ratio, resizes to a target dimension,
    and saves these processed cropped images to CROPPED_OUTPUT_DIR.
    It also generates a metadata JSON file tracking the process and results.
    """
    print("--- Starting Enhanced Person Extraction Process ---")
    print(f"Model: {MODEL_NAME}, Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(
        f"Crop Padding Ratio: {CROP_PADDING_RATIO}, "
        f"Target Classifier Size: {TARGET_CLASSIFIER_SIZE}x{TARGET_CLASSIFIER_SIZE}px")
    print(f"Minimum Crop Dimension: {MIN_CROP_DIMENSION_PX}px")

    # --- Ensure output directories exist ---
    try:
        os.makedirs(CROPPED_OUTPUT_DIR, exist_ok=True)
        os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)
        print(f"Ensured '{CROPPED_OUTPUT_DIR}' and '{INFERENCE_OUTPUT_DIR}' exist.")
    except OSError as e:
        print(f"Error creating output directories: {e}. Exiting.")
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

    # --- Gather Image Files ---
    image_files=[os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in '{IMAGE_DIR}'. Please add some images.")
        return

    print(f"Found {len(image_files)} images to process in '{IMAGE_DIR}'.")

    total_persons_extracted=0
    all_crops_metadata=[]  # List to store metadata for all generated crops

    try:
        # --- Run Inference ---
        results=model.predict(source=image_files,
                              save=True,
                              project=INFERENCE_OUTPUT_DIR,
                              name='yolov12_person_extraction_for_classifier',
                              conf=CONFIDENCE_THRESHOLD,
                              verbose=False)

        # --- Process Each Image's Results ---
        for i, result in enumerate(results):
            image_path=image_files[i]
            image_filename=os.path.basename(image_path)
            print(f"\n--- Processing detections for: {image_filename} ---")

            try:
                original_image=Image.open(image_path).convert("RGB")
            except Exception as img_e:
                print(f"  Error loading original image '{image_filename}': {img_e}. Skipping image.")
                continue

            person_count_in_image=0

            if result.boxes:
                for j, box in enumerate(result.boxes):
                    class_id=int(box.cls[0])
                    label=model.names[class_id]
                    confidence=float(box.conf[0])
                    x1_orig, y1_orig, x2_orig, y2_orig=box.xyxy[0].tolist()

                    print(
                        f"  - Detected: {label} (Confidence: {confidence:.2f}) [BBox: [{x1_orig:.2f}, {y1_orig:.2f}, {x2_orig:.2f}, {y2_orig:.2f}]]")

                    # --- Extract and Save 'person' Crops ---
                    if label=='person':
                        # 1. Apply Padding to Bounding Box
                        bbox_width=x2_orig-x1_orig
                        bbox_height=y2_orig-y1_orig

                        pad_x=bbox_width*CROP_PADDING_RATIO/2
                        pad_y=bbox_height*CROP_PADDING_RATIO/2

                        x1_padded=max(0, int(x1_orig-pad_x))
                        y1_padded=max(0, int(y1_orig-pad_y))
                        x2_padded=min(original_image.width, int(x2_orig+pad_x))
                        y2_padded=min(original_image.height, int(y2_orig+pad_y))

                        # 2. Filter by Minimum Size
                        current_crop_width=x2_padded-x1_padded
                        current_crop_height=y2_padded-y1_padded

                        if current_crop_width<MIN_CROP_DIMENSION_PX or current_crop_height<MIN_CROP_DIMENSION_PX:
                            print(
                                f"    Skipping small person detection (dim: {current_crop_width}x{current_crop_height}px), below min {MIN_CROP_DIMENSION_PX}px.")
                            continue

                        # 3. Perform Initial Crop with Padding
                        try:
                            padded_crop=original_image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
                        except Exception as crop_e:
                            print(f"    Error during initial crop for person {j}: {crop_e}. Skipping.")
                            continue

                        # 4. Pad to Square Aspect Ratio and Resize (using utility function)
                        final_cropped_image=pad_to_square_and_resize(padded_crop, TARGET_CLASSIFIER_SIZE)

                        # 5. Save the Final Cropped Image
                        base_name=os.path.splitext(image_filename)[0]
                        cropped_filename=f"{base_name}_person_{j}_conf{confidence:.2f}_{TARGET_CLASSIFIER_SIZE}px.jpg"
                        cropped_filepath=os.path.join(CROPPED_OUTPUT_DIR, cropped_filename)

                        try:
                            final_cropped_image.save(cropped_filepath)
                            print(f"    -> Saved processed person crop to: {cropped_filepath}")
                            person_count_in_image+=1
                            total_persons_extracted+=1

                            # 6. Store Metadata for this crop
                            crop_id=f"{base_name}_person_{j}_conf{confidence:.4f}".replace('.', '_')

                            metadata_entry={
                                "crop_id": crop_id,
                                "original_image_filename": image_filename,
                                "original_image_path": os.path.abspath(image_path),
                                "yolo_model_used": MODEL_NAME,
                                "yolo_confidence_score": confidence,
                                "original_bbox_xyxy": [round(val, 2) for val in [x1_orig, y1_orig, x2_orig, y2_orig]],
                                "padded_bbox_xyxy": [x1_padded, y1_padded, x2_padded, y2_padded],
                                "original_crop_dimensions_wh": [padded_crop.width, padded_crop.height],
                                # Dimensions before square padding/resize
                                "final_crop_filename": cropped_filename,
                                "final_crop_path": os.path.abspath(cropped_filepath),
                                "final_crop_resolution_wh": [TARGET_CLASSIFIER_SIZE, TARGET_CLASSIFIER_SIZE],
                                "cropping_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                                    timespec='seconds'),
                                "min_crop_dimension_filter_applied": MIN_CROP_DIMENSION_PX,
                                "crop_padding_ratio_applied": CROP_PADDING_RATIO,
                                "status": "raw_crop_generated",
                                "assigned_label": None,
                                "label_source": None,
                                "labeling_timestamp": None,
                                "dataset_split": None
                            }
                            all_crops_metadata.append(metadata_entry)

                        except Exception as save_e:
                            print(f"    Error saving processed image '{cropped_filename}': {save_e}")
            else:
                print("  No objects detected in this image.")

            if person_count_in_image==0:
                print(
                    f"  No 'person' objects (above confidence {CONFIDENCE_THRESHOLD} and min size {MIN_CROP_DIMENSION_PX}px) detected in {image_filename}.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during inference or extraction: {e}")
        if "No module named 'PIL'" in str(e):
            print("Hint: The 'Pillow' library is required for image manipulation. Install it with: pip install Pillow")
        print(
            "Hint: Check your image files for corruption or unsupported formats if encountering image-related errors.")

    print(f"\n--- Enhanced Extraction Process Complete ---")
    print(f"Total 'person' images extracted: {total_persons_extracted}")
    print(f"Processed cropped images can be found in: '{CROPPED_OUTPUT_DIR}'")
    print(
        f"Annotated input images saved in: '{os.path.join(INFERENCE_OUTPUT_DIR, 'yolov12_person_extraction_for_classifier')}'")

    # --- Save Metadata to JSON File (using utility function) ---
    metadata_filepath=os.path.join(CROPPED_OUTPUT_DIR, METADATA_FILENAME)
    try:
        save_metadata(all_crops_metadata, metadata_filepath)
        print(f"Detailed metadata for all crops saved to: '{metadata_filepath}'")
    except Exception as json_e:
        print(f"Error saving metadata to JSON file '{metadata_filepath}': {json_e}")


if __name__=="__main__":
    extract_and_store_persons_for_classification()
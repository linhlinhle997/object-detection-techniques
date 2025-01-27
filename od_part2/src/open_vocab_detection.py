import uuid
import cv2
import os
import argparse
from ultralytics import YOLOWorld
from ultralytics.engine.results import Boxes
from src.utils import setup_logger


def save_detection_results(results, image_path, output_dir, logger):
    """Save detection results as images if detections were found."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, result in enumerate(results):
        # Skip if no objects detected
        if not len(result.boxes):
            logger.info(f"No detections found in result {i + 1}")
            continue

        logger.info(f"Detections found in result {i + 1}.")
        
        # Generate visualization of detections
        annotated_image = result.plot()

        # Create unique filename and save the image
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{image_name}_ovd.jpg")
        cv2.imwrite(output_path, annotated_image)
        saved_paths.append(str(output_path))
        
        logger.info(f"Image saved to {output_path}")
    return saved_paths


def main():
    logger = setup_logger()
    
    parser = argparse.ArgumentParser(description="Video Object Tracking")
    parser.add_argument("--image-path", type=str, default="data/vietnam_3.jpg", help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save the output image")
    parser.add_argument("--model-path", type=str, default="yolov8x-world.pt", help="Path to YOLO model")
    args = parser.parse_args()
    
    try:                
        # Load and configure YOLO model
        logger.info(f"Loading YOLO model from: {args.model_path}")
        model = YOLOWorld(args.model_path)
        model.set_classes(["mask", "glasses"]) # Define custom classes
        
        # Perform object detection
        logger.info(f"Running predictions on image: {args.image_path}")
        results: Boxes = model.predict(args.image_path)
        save_detection_results(results, args.image_path, args.output_dir, logger)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        return


if __name__ == "__main__":
    main()

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from src.utils import setup_logger, get_video_properties, prepare_video_writer


def draw_tracking_lines(frame, track_history, boxes, track_ids, track_history_length):
    """Visualize the movement paths of tracked objects on the frame."""
    for box, track_id in zip(boxes, track_ids):
        # Extract the center coordinates of the bounding box
        x, y, _, _ = box
        track = track_history[track_id]
        track.append((float(x), float(y)))

        # Limit the length of the track history to prevent excessive memory usage
        if len(track) > track_history_length:
            track.pop(0)
        
        # Draw the movement path
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=4)
    return frame


def process_video(video_path, output_dir, model_path):
    """Process a video file using YOLO object tracking."""
    logger = setup_logger()
    
    try:
        model = YOLO(model_path) # Load the YOLO model
        cap = cv2.VideoCapture(video_path) # Open the video
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return

        # Get video properties
        width, height, fps, total_frames = get_video_properties(cap)
        logger.info(f"Video properties: Width={width}, Height={height}, FPS={fps}, Total Frames={total_frames}")

        # Prepare output file
        out, output_path = prepare_video_writer(output_dir, video_path, "tracked_simple", fps, width, height)

        # Initialize tracking
        track_history = defaultdict(list)
        track_history_length = 120 # Maximum number of frames to keep in track history

        logger.info("Processing video...")
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if not success:
                break

            # Perform object detection and tracking
            # 'persist=True' maintains tracking across frames
            results = model.track(frame, persist=True, show=False)
            
            # Extract bounding boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            
            # Annotate frame with object detections and tracking paths
            annotated_frame = results[0].plot(font_size=4, line_width=2)
            annotated_frame = draw_tracking_lines(annotated_frame, track_history, boxes, track_ids, track_history_length)

            # Write annotated frame to output video
            out.write(annotated_frame)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
    finally:
        # Clean up resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Output video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Video Object Tracking")
    parser.add_argument("--video-path", type=str, default="data/vietnam.mp4", help="Path to input video")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save the output video")
    parser.add_argument("--model-path", type=str, default="yolo11l.pt", help="Path to YOLO model")
    args = parser.parse_args()

    process_video(args.video_path, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()

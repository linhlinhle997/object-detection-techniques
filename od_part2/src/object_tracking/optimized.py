import argparse
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from ultralytics import YOLO
from src.utils import setup_logger, get_video_properties, prepare_video_writer, save_batch_as_images


def update_track_history(track_history, last_seen, current_tracks, frame_count, frame_idx, track_history_length):
    """Manage the history of tracked objects and remove inactive tracks."""
    # Remove tracks that haven't been seen for longer than the specified history length
    for track_id in list(track_history.keys()):
        if track_id in current_tracks:
            # Update the last seen frame for active tracks
            last_seen[track_id] = frame_count - (len(track_history) - frame_idx - 1)
        elif frame_count - last_seen[track_id] > track_history_length:
            # Remove tracks that have been inactive for too long
            del track_history[track_id]
            del last_seen[track_id]


def draw_track_history(frame, boxes, track_ids, track_history, track_history_length):
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


def process_batch(batch_frames, model, track_history, last_seen, track_history_length, frame_count):
    """Process a batch of frames and return the processed frames."""
    results = model.track(batch_frames, persist=True, tracker="botsort.yaml", verbose=False, iou=0.5)
    processed_frames = []

    for frame_idx, result in enumerate(results):
        # Extract bounding boxes and track IDs
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []

        # Update and manage track history
        current_tracks = set(track_ids)
        update_track_history(track_history, last_seen, current_tracks, frame_count, frame_idx, track_history_length)

        # Annotate frame with detection boxes and tracking information
        annotated_frame = result.plot(font_size=4, line_width=2)
        annotated_frame = draw_track_history(annotated_frame, boxes, track_ids, track_history, track_history_length)

        processed_frames.append(annotated_frame)

    return processed_frames


def process_video(video_path, output_dir, model_path):
    """Process a video file using YOLO object tracking."""
    logger = setup_logger()
    
    try:
        model = YOLO(model_path) # Load YOLO model
        cap = cv2.VideoCapture(video_path) # Open video file
        if not cap.isOpened():
            logger.error(f"Failed to open video {video_path}")
            return
        
        # Get video details
        width, height, fps, total_frames = get_video_properties(cap)
        logger.info(f"Video properties: Width={width}, Height={height}, FPS={fps}, Total Frames={total_frames}")
        
        # Prepare output file
        out, output_path = prepare_video_writer(output_dir, video_path, "tracked_optimized", fps, width, height)

        # Initialize tracking
        track_history = defaultdict(list) # Store movement history for each tracked object
        last_seen = defaultdict(int) # Track the last frame each object was seen
        batch_size = 64 # Number of frames to process in each batch
        track_history_length = 120 # Maximum number of frames to keep in track history
        
        logger.info("Starting video processing.")
        with tqdm(total=total_frames, desc="Processing frames", colour="green") as pbar:
            frame_count = 0
            batch_frames = []
            
            # Iterate through video frames
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                batch_frames.append(frame) # Collect frames for batch processing
                
                # Process batches
                if len(batch_frames) == batch_size or frame_count == total_frames:
                    processed_frames = process_batch(batch_frames, model, track_history, last_seen, track_history_length, frame_count)
                    
                    # Save the batch of processed frames as images
                    # save_batch_as_images(output_dir, processed_frames, frame_count)
                    
                    # Save frames to output video
                    for frame in processed_frames:
                        out.write(frame)
                        pbar.update(1)
                    
                    batch_frames = []
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

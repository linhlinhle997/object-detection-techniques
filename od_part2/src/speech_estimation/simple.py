import cv2 
import argparse
from ultralytics import solutions
from src.utils import setup_logger, prepare_video_writer, get_video_properties


def process_video(video_path, output_dir, model_path):
    """Process video for speed estimation."""
    logger = setup_logger()
    
    try:
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        # Get video properties
        width, height, fps, total_frames = get_video_properties(cap)
        logger.info(f"Video properties: Width={width}, Height={height}, FPS={fps}, Total Frames={total_frames}")

        # thai.mp4 - region points
        speed_region = [
            (int(width*0.0), int(height*0.7)), # Top left
            (width, int(height*0.7)), # Top right
            (width, int(height*0.9)), # Bottom right
            (int(width*0.0), int(height*0.9)) # Bottom left
        ]
        
        # Prepare output video
        out, output_path = prepare_video_writer(output_dir, video_path, "speedest_simple", fps, width, height)
        
        # Init speed estimator
        speed = solutions.SpeedEstimator(show=False, model=model_path, region=speed_region)

        frame_count = 0
        
        # Process video frames
        logger.info("Starting video processing.")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Estimate speed
            speed.estimate_speed(frame)
            
            # Write frame
            out.write(frame)
            
        logger.info(f"Processed {frame_count} frames successfully")
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
    parser.add_argument("--video-path", type=str, default="data/thai.mp4", help="Path to input video")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save the output video")
    parser.add_argument("--model-path", type=str, default="yolo11n.pt", help="Path to YOLO model")
    args = parser.parse_args()
    
    process_video(args.video_path, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()

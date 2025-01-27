import cv2
import argparse
from ultralytics import solutions
from tqdm import tqdm
from src.utils import setup_logger, get_video_properties, prepare_video_writer, save_batch_as_images


def process_batch(counter, batch_frames):
    """Process a batch of frames through the object counter."""
    try:
        return [counter.count(frame) for frame in batch_frames]
    except Exception as e:
        raise RuntimeError(f"Error processing batch: {str(e)}")


def process_video(video_path, output_dir, model_path, region_points):
    """Process video for object counting"""
    logger = setup_logger()
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        width, height, fps, total_frames = get_video_properties(cap)
        logger.info(f"Video properties: Width={width}, Height={height}, FPS={fps}, Total Frames={total_frames}")
        
        # Prepare output video
        out, output_path = prepare_video_writer(output_dir, video_path, "counted_optimized", fps, width, height)
        
        # Initialize Object Counter
        counter = solutions.ObjectCounter(show=False, region=region_points, model=model_path)
        
        # Process video frames
        batch_size = 64
        batch_frames = []
        frame_count = 0
        
        logger.info("Starting video processing.")
        with tqdm(total=total_frames, desc="Processing frames", colour="green") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Collect frames for batch processing
                frame_count += 1
                batch_frames.append(frame) 
                
                # Process when batch is full or at end of video
                if len(batch_frames) == batch_size or frame_count == total_frames:
                    processed_frames = process_batch(counter, batch_frames)
                    
                    # Save the batch of processed frames as images
                    # save_batch_as_images(output_dir, processed_frames, frame_count)

                    for processed_frame in processed_frames:
                        out.write(processed_frame)
                        pbar.update(1)

                    batch_frames = []
                
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
    parser.add_argument("--video-path", type=str, default="data/highway.mp4", help="Path to input video")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save the output video")
    parser.add_argument("--model-path", type=str, default="yolo11x.pt", help="Path to YOLO model")
    args = parser.parse_args()
    
    # highway.mp4 - region points
    region_points = [
        (430, 700),   # top left
        (1600, 700),  # top right
        (1600, 1080), # bottom right
        (430, 1080)   # bottom left
    ]
    
    # highway-2.mp4 - region points
    # region_points = [
    #     (0, 600),   # top left
    #     (3300, 600),  # top right
    #     (3300, 1080), # bottom right
    #     (0, 1080)   # bottom left
    # ]
    
    # fruit_and_vegetable.gif - region points
    # region_points = [(250, 0), (250, 270)]
    
    process_video(args.video_path, args.output_dir, args.model_path, region_points)


if __name__ == "__main__":
    main()

import logging
import cv2
import os


def setup_logger():
    """Set up the logger for debugging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def get_video_properties(cap):
    """Extract video properties from capture object."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, total_frames


def prepare_video_writer(output_dir, video_path, suffix, fps, width, height):
    """Prepares a cv2.VideoWriter object for saving the output video."""
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_{suffix}.mp4")
    writer = cv2.VideoWriter(
        output_path, 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps, 
        (width, height)
    )
    return writer, output_path


def save_batch_as_images(output_dir, batch_frames, start_index):
    """Save a batch of frames as images."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(batch_frames):
            filename = os.path.join(output_dir, f"frame_{start_index + i:05d}.jpg")
            cv2.imwrite(filename, frame)
    except Exception as e:
        raise RuntimeError(f"Error saving batch of frames: {str(e)}")

    
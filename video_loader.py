# video_loader.py

import cv2
import os

def load_and_sample_video(video_path, frame_interval=10, max_frames=None):
    """
    Loads an mp4 video and samples frames at a regular interval.

    Args:
        video_path (str): Path to the .mp4 video file.
        frame_interval (int): Sample one frame every `frame_interval` frames.
        max_frames (int or None): Optional limit on number of frames to return.

    Returns:
        frames (List[np.ndarray]): List of sampled frames as BGR images (OpenCV format).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break

        count += 1

    cap.release()
    return frames

# Example usage
if __name__ == "__main__":
    video_file = "sample.mp4"  # Place an mp4 video with this name in the same folder
    sampled_frames = load_and_sample_video(video_file, frame_interval=15, max_frames=20)
    print(f"Sampled {len(sampled_frames)} frames from {video_file}")

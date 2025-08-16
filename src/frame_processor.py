# frame_processor.py
"""
Frame processing functions for the motion detection project.
"""

import cv2
import numpy as np


def process_video(video_path, target_fps=5, resize_dim=(1280, 720)):
    """
    Extract frames from a video at a specified frame rate.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second to extract
        resize_dim: Dimensions to resize frames to (width, height)

    Returns:
        List of extracted frames
    """
    frames = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {original_fps}, Total frames: {total_frames}")
    print(f"Target FPS: {target_fps}")
    
    # Calculate frame interval for target FPS
    frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Resize frame
            resized_frame = cv2.resize(frame, resize_dim)
            frames.append(resized_frame)
            extracted_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from {frame_count} total frames")
    
    return frames


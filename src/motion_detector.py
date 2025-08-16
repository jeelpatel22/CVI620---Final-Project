# motion_detector.py
"""
Motion detection functions for the sports video analysis project.
"""

import cv2
import numpy as np

def detect_motion(frames, frame_idx, threshold=25, min_area=100):
    """
    Detect motion in the current frame by comparing with previous frame.

    Args:
        frames: List of video frames
        frame_idx: Index of the current frame
        threshold: Threshold for frame difference detection
        min_area: Minimum contour area to consider

    Returns:
        List of bounding boxes for detected motion regions
    """
    # We need at least 2 frames to detect motion
    if frame_idx < 1 or frame_idx >= len(frames):
        return []

    # Get current and previous frame
    current_frame = frames[frame_idx]
    prev_frame = frames[frame_idx - 1]

    # Convert frames to grayscale
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
    gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)
    
    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(gray_prev, gray_current)
    
    # Apply threshold to highlight differences
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and extract bounding boxes
    motion_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))
    
    return motion_boxes
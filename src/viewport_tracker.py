# viewport_tracker.py
"""
Viewport tracking functions for creating a smooth "virtual camera".
"""

import cv2
import numpy as np


def calculate_region_of_interest(motion_boxes, frame_shape):
    """
    Calculate the primary region of interest based on motion boxes.

    Args:
        motion_boxes: List of motion detection bounding boxes
        frame_shape: Shape of the video frame (height, width)

    Returns:
        Tuple (x, y, w, h) representing the region of interest center point and dimensions
    """
    if not motion_boxes:
        # If no motion is detected, use the center of the frame
        height, width = frame_shape[:2]
        return (width // 2, height // 2, 0, 0)

    # Strategy: Use weighted average of all motion boxes
    # Larger boxes get more weight in determining the center
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    
    for x, y, w, h in motion_boxes:
        # Calculate center of the box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Use area as weight (larger motion regions are more important)
        weight = w * h
        
        weighted_x += center_x * weight
        weighted_y += center_y * weight
        total_weight += weight
    
    if total_weight > 0:
        # Calculate weighted average center
        avg_x = int(weighted_x / total_weight)
        avg_y = int(weighted_y / total_weight)
        
        # Return center coordinates and dimensions (using largest box for reference)
        largest_box = max(motion_boxes, key=lambda box: box[2] * box[3])
        return (avg_x, avg_y, largest_box[2], largest_box[3])
    else:
        height, width = frame_shape[:2]
        return (width // 2, height // 2, 0, 0)


def track_viewport(frames, motion_results, viewport_size, smoothing_factor=0.3):
    """
    Track viewport position across frames with smoothing.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_size: Tuple (width, height) of the viewport
        smoothing_factor: Factor for smoothing viewport movement (0-1)
                          Lower values create smoother movement

    Returns:
        List of viewport positions for each frame as (x, y) center coordinates
    """
    viewport_positions = []

    # Initialize with center of first frame if available
    if not frames:
        return []
    
    height, width = frames[0].shape[:2]
    vp_width, vp_height = viewport_size
    
    # Initialize previous position with frame center
    prev_x, prev_y = width // 2, height // 2

    for i, motion_boxes in enumerate(motion_results):
        # Calculate region of interest for current frame
        roi_x, roi_y, _, _ = calculate_region_of_interest(motion_boxes, frames[i].shape)
        
        # Apply smoothing using exponential moving average
        if i == 0:
            # For first frame, use ROI directly
            smooth_x = roi_x
            smooth_y = roi_y
        else:
            # Smooth the movement
            smooth_x = int(prev_x * (1 - smoothing_factor) + roi_x * smoothing_factor)
            smooth_y = int(prev_y * (1 - smoothing_factor) + roi_y * smoothing_factor)
        
        # Ensure viewport stays within frame boundaries
        # Adjust X coordinate
        min_x = vp_width // 2
        max_x = width - vp_width // 2
        smooth_x = max(min_x, min(max_x, smooth_x))
        
        # Adjust Y coordinate
        min_y = vp_height // 2
        max_y = height - vp_height // 2
        smooth_y = max(min_y, min(max_y, smooth_y))
        
        # Store viewport center position
        viewport_positions.append((smooth_x, smooth_y))
        
        # Update previous position for next iteration
        prev_x, prev_y = smooth_x, smooth_y

    return viewport_positions

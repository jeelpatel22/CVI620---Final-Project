# visualizer.py
"""
Visualization functions for displaying motion detection and viewport tracking results.
"""

import os
import cv2
import numpy as np


def visualize_results(frames, motion_results, viewport_positions, viewport_size, output_dir):
    """
    Create visualization of motion detection and viewport tracking results.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_positions: List of viewport center positions for each frame
        viewport_size: Tuple (width, height) of the viewport
        output_dir: Directory to save visualization results
    """
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    viewport_dir = os.path.join(output_dir, "viewport")
    os.makedirs(viewport_dir, exist_ok=True)

    # Get dimensions for the output video
    height, width = frames[0].shape[:2]

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "motion_detection.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 5, (width, height))

    viewport_video_path = os.path.join(output_dir, "viewport_tracking.mp4")
    vp_width, vp_height = viewport_size
    viewport_writer = cv2.VideoWriter(
        viewport_video_path, fourcc, 5, (vp_width, vp_height)
    )

    for i, frame in enumerate(frames):
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Get motion boxes and viewport position for this frame
        motion_boxes = motion_results[i] if i < len(motion_results) else []
        vp_x, vp_y = viewport_positions[i] if i < len(viewport_positions) else (width//2, height//2)
        
        # Draw bounding boxes around motion regions (green color)
        for x, y, w, h in motion_boxes:
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate viewport rectangle coordinates
        vp_left = vp_x - vp_width // 2
        vp_top = vp_y - vp_height // 2
        vp_right = vp_x + vp_width // 2
        vp_bottom = vp_y + vp_height // 2
        
        # Ensure viewport coordinates are within frame bounds
        vp_left = max(0, vp_left)
        vp_top = max(0, vp_top)
        vp_right = min(width, vp_right)
        vp_bottom = min(height, vp_bottom)
        
        # Draw the viewport rectangle (blue color)
        cv2.rectangle(vis_frame, (vp_left, vp_top), (vp_right, vp_bottom), (255, 0, 0), 3)
        
        # Extract the viewport content
        viewport_frame = frame[vp_top:vp_bottom, vp_left:vp_right]
        
        # Resize viewport to exact viewport size if needed
        if viewport_frame.shape[:2] != (vp_height, vp_width):
            viewport_frame = cv2.resize(viewport_frame, (vp_width, vp_height))
        
        # Add frame number to the visualization
        cv2.putText(vis_frame, f"Frame {i+1}/{len(frames)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(viewport_frame, f"Frame {i+1}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization frames as images
        frame_filename = os.path.join(frames_dir, f"frame_{i+1:04d}.jpg")
        cv2.imwrite(frame_filename, vis_frame)
        
        viewport_filename = os.path.join(viewport_dir, f"viewport_{i+1:04d}.jpg")
        cv2.imwrite(viewport_filename, viewport_frame)
        
        # Write frames to both video writers
        video_writer.write(vis_frame)
        viewport_writer.write(viewport_frame)
    
    # Release the video writers
    video_writer.release()
    viewport_writer.release()

    print(f"Visualization saved to {video_path}")
    print(f"Viewport video saved to {viewport_video_path}")
    print(f"Individual frames saved to {frames_dir} and {viewport_dir}")
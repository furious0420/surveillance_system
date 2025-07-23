#!/usr/bin/env python3
"""
Create a test aerial video for demonstration
"""

import cv2
import numpy as np
import os

def create_test_aerial_video(output_path="test_aerial_video.mp4", duration_seconds=10, fps=30):
    """Create a synthetic aerial video with moving objects"""
    
    # Video properties
    width, height = 1280, 720
    total_frames = duration_seconds * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration_seconds}s, FPS: {fps}, Frames: {total_frames}")
    
    for frame_num in range(total_frames):
        # Create base aerial background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add terrain background (green)
        frame[:, :] = [34, 139, 34]  # Forest green
        
        # Add some static infrastructure
        # Roads
        cv2.rectangle(frame, (0, height//2-20), (width, height//2+20), (64, 64, 64), -1)
        cv2.rectangle(frame, (width//2-20, 0), (width//2+20, height), (64, 64, 64), -1)
        
        # Buildings (static)
        buildings = [
            (200, 150, 300, 250),
            (500, 100, 650, 200),
            (800, 300, 950, 400),
            (100, 450, 200, 550),
            (1000, 200, 1150, 300)
        ]
        
        for x1, y1, x2, y2 in buildings:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), -1)
            # Add some detail
            cv2.rectangle(frame, (x1+10, y1+10), (x2-10, y2-10), (96, 96, 96), 2)
        
        # Add moving vehicles
        progress = frame_num / total_frames
        
        # Vehicle 1: Moving horizontally
        vehicle1_x = int(50 + (width - 100) * progress)
        vehicle1_y = height//2 - 10
        cv2.rectangle(frame, (vehicle1_x, vehicle1_y), (vehicle1_x+20, vehicle1_y+10), (255, 255, 255), -1)
        
        # Vehicle 2: Moving vertically
        vehicle2_x = width//2 - 10
        vehicle2_y = int(50 + (height - 100) * progress)
        cv2.rectangle(frame, (vehicle2_x, vehicle2_y), (vehicle2_x+10, vehicle2_y+20), (0, 0, 255), -1)
        
        # Vehicle 3: Moving diagonally
        vehicle3_x = int(100 + (width - 200) * progress)
        vehicle3_y = int(100 + (height - 200) * progress)
        cv2.rectangle(frame, (vehicle3_x, vehicle3_y), (vehicle3_x+15, vehicle3_y+15), (255, 0, 0), -1)
        
        # Add some circular objects (storage tanks)
        tanks = [
            (400, 400, 30),
            (700, 500, 25),
            (900, 150, 35)
        ]
        
        for x, y, radius in tanks:
            cv2.circle(frame, (x, y), radius, (139, 69, 19), -1)
            cv2.circle(frame, (x, y), radius-5, (160, 82, 45), 2)
        
        # Add some sports facilities
        # Tennis court
        cv2.rectangle(frame, (600, 250), (750, 350), (0, 128, 0), -1)
        cv2.rectangle(frame, (600, 250), (750, 350), (255, 255, 255), 2)
        cv2.line(frame, (675, 250), (675, 350), (255, 255, 255), 2)
        
        # Basketball court
        cv2.rectangle(frame, (300, 500), (450, 600), (139, 69, 19), -1)
        cv2.circle(frame, (375, 550), 30, (255, 255, 255), 2)
        
        # Add frame number for reference
        cv2.putText(frame, f"Frame: {frame_num+1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add some noise for realism
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.95, noise, 0.05, 0)
        
        # Write frame
        out.write(frame)
        
        if (frame_num + 1) % 30 == 0:
            print(f"Generated {frame_num + 1}/{total_frames} frames ({(frame_num+1)/total_frames*100:.1f}%)")
    
    # Release video writer
    out.release()
    
    print(f"Test video created successfully: {output_path}")
    print(f"Video size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return output_path

if __name__ == "__main__":
    # Create test video
    video_path = create_test_aerial_video()
    
    print(f"\nTest video ready! You can now run:")
    print(f"python video_inference.py yolo11s-obb.pt --input {video_path} --output annotated_output.mp4")
    print(f"python video_inference.py yolo11s-obb.pt --input {video_path}")

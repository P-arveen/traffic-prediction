import cv2
import numpy as np

def display_dashboard(frames):
    if len(frames) != 4:
        # Fallback if not exactly 4 frames
        for i, f in enumerate(frames):
            cv2.imshow(f"Lane {i+1}", f)
        return

    # Resize frames to a uniform size before concatenating (e.g., 640x360 each)
    resized_frames = [cv2.resize(f, (640, 360)) for f in frames]
    
    top_row = np.hstack((resized_frames[0], resized_frames[1]))
    bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
    
    dashboard = np.vstack((top_row, bottom_row))
    
    # Add a global title bar
    title_bar = np.zeros((50, dashboard.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, "SMART TRAFFIC MANAGEMENT DASHBOARD (MAJOR PROJECT)", 
                (200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    dashboard_with_title = np.vstack((title_bar, dashboard))
    
    cv2.imshow("Smart Traffic Control Center", dashboard_with_title)

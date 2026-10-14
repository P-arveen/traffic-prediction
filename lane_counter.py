import cv2
import numpy as np
from tracker import CentroidTracker

# Polygon ROI definition for realistic intersection counting
def get_roi(w, h):
    return np.array([
        [int(w * 0.1), int(h * 0.4)],
        [int(w * 0.9), int(h * 0.4)],
        [w, h],
        [0, h]
    ], np.int32)

def is_emergency_vehicle(frame, box):
    # Heuristic: Check for high presence of bright red/blue or white (flashing lights / ambulance body)
    x1, y1, x2, y2 = map(int, box[:4])
    # clamp coords
    h_f, w_f = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_f, x2), min(h_f, y2)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return False
        
    # Convert to HSV to find red/blue
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Red has two ranges in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Blue range
    lower_blue = np.array([100, 150, 150])
    upper_blue = np.array([140, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    mask_total = mask_red1 + mask_red2 + mask_blue
    red_blue_ratio = cv2.countNonZero(mask_total) / (crop.shape[0] * crop.shape[1] + 1e-6)
    
    return red_blue_ratio > 0.05 # If > 5% of pixels are bright red/blue

class LaneCounter:
    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.vehicle_count = 0
        self.tracker = CentroidTracker(max_disappeared=15, max_distance=60)
        self.roi = None
        self.tracked_objects = {}
        self.has_emergency = False
        self.has_accident = False
        self.red_light_jumps = 0
        self.total_idle_frames = 0

    def is_in_roi(self, box):
        if self.roi is None:
            return True
        # center of box
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        # check if in polygon
        return cv2.pointPolygonTest(self.roi, (cx, cy), False) >= 0

    def count_vehicles(self, boxes, frame, is_green):
        h, w = frame.shape[:2]
        self.roi = get_roi(w, h)
        
        ALLOWED_CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
        
        rects = []
        labels = []
        emergency_detected = False
        for box in boxes:
            if box[-1] in ALLOWED_CLASSES and self.is_in_roi(box):
                rects.append(box[:4])
                
                # Check for emergency
                is_emerg = False
                if box[-1] in ['truck', 'bus', 'car']:
                    is_emerg = is_emergency_vehicle(frame, box)
                    if is_emerg:
                        emergency_detected = True
                        
                labels.append('Emergency' if is_emerg else box[-1])
                
        self.tracked_objects = self.tracker.update(rects, labels)
        self.vehicle_count = len(self.tracked_objects)
        self.has_emergency = emergency_detected
        
        # Accident and Rules logic
        accident_flag = False
        for obj_id, info in self.tracked_objects.items():
            # Accident: Idling for 90+ frames (~3 sec) while light is GREEN
            if is_green and info.get('idle_frames', 0) > 90:
                accident_flag = True
                
            # Red Light Jump: centroid Y > 85% of frame while RED
            if not is_green:
                cy = info['centroid'][1]
                # To trigger only once per vehicle, we can use a flag. 
                # We'll just keep it simple: if it just crossed
                if cy > int(h * 0.85) and not info.get('jump_counted', False):
                    self.red_light_jumps += 1
                    info['jump_counted'] = True
                    
        self.has_accident = accident_flag
        
        self.total_idle_frames = sum([obj.get('idle_frames', 0) for obj in self.tracked_objects.values()])
        
        # Vehicle Type Weighting for schedule
        weighted_score = 0
        for info in self.tracked_objects.values():
            lbl = info['label']
            if lbl in ['bicycle', 'motorcycle']:
                weighted_score += 0.5
            elif lbl in ['truck', 'bus', 'Emergency']:
                weighted_score += 2.5
            else:
                weighted_score += 1.0
                
        return weighted_score

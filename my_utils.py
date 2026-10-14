import cv2

def enhance_frame(frame):
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Check average brightness
    avg_brightness = l_channel.mean()
    if avg_brightness < 100: # It's a low-light or hazy frame
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # Merge back
        limg = cv2.merge((cl, a, b))
        final_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final_frame
    return frame

# Convert BGR to RGB for YOLO
def preprocess_frame(frame):
    enhanced = enhance_frame(frame)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

# Extract usable detections (format: [x1, y1, x2, y2, class_name])
def postprocess_detections(results):
    boxes = results.xyxy[0].cpu().numpy()
    names = results.names
    return [
        [int(b[0]), int(b[1]), int(b[2]), int(b[3]), names[int(b[5])]]
        for b in boxes if b[4] > 0.4
    ]

# State for anti-starvation
waiting_times = [0, 0, 0, 0]

# Dynamic scheduling based on lane count and wait time
def schedule_lights(counts, emergencies, current_green_idx):
    global waiting_times
    
    # Update waiting times
    for i in range(len(counts)):
        if i == current_green_idx:
            waiting_times[i] = 0
        else:
            if counts[i] > 0:
                waiting_times[i] += 1
            else:
                waiting_times[i] = 0
                
    # Calculate scores: 1 point per vehicle, 0.5 points per frame waited
    scores = []
    for i in range(len(counts)):
        # If emergency, score is infinity effectively
        if emergencies[i]:
            scores.append(float('inf'))
        else:
            scores.append(counts[i] * 1.0 + waiting_times[i] * 0.5)
            
    # Which lane should be green now?
    best_lane = scores.index(max(scores)) if (sum(scores) > 0 or max(scores) == float('inf')) else current_green_idx
    
    # Calculate durations (standard map)
    times = []
    total = sum(counts)
    if total == 0:
        times = [15] * len(counts)
    else:
        for c in counts:
            if c == 0:
                times.append(0)
            else:
                time = int((c / total) * 60)
                times.append(max(5, time)) # Minimum 5s green light
                
    # Force min green for emergency
    if emergencies[best_lane]:
        times[best_lane] = max(times[best_lane], 20)
        
    return times, best_lane

# Overlay result info on frame
def draw_results(frame, counter, duration, is_green):
    # Draw ROI
    if counter.roi is not None:
        cv2.polylines(frame, [counter.roi], True, (255, 0, 0), 2)

    # Draw Tracked Objects
    for obj_id, info in counter.tracked_objects.items():
        box = info['box']
        label = info['label']
        centroid = info['centroid']
        idle = info.get('idle_frames', 0)
        
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        txt = f"[ID: {obj_id}] {label}"
        if idle > 30: # 1 second idle
            txt += f" (Idle)"
        
        col = (0, 0, 255) if label == 'Emergency' else (0, 255, 0)
        cv2.putText(frame, txt, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

    # Draw Traffic Light
    h, w = frame.shape[:2]
    light_color = (0, 255, 0) if is_green else (0, 0, 255)
    light_text = "GREEN" if is_green else "RED"
    
    cv2.circle(frame, (w - 40, 40), 20, light_color, -1)
    cv2.putText(frame, light_text, (w - 65, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)

    # Overlay Text
    co2_g = (counter.total_idle_frames / 30.0) * 0.05 
    
    text1 = f"Lane {counter.lane_id + 1}: {counter.vehicle_count} veh | Time: {duration}s"
    text2 = f"CO2: {co2_g:.2f}g | Emerg: {'YES' if counter.has_emergency else 'NO'}"
    
    cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

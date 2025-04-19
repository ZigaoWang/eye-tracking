import cv2
import numpy as np
import os
import pyautogui
import time
from collections import deque
import pygame  # Added for audio playback

# --- Constants ---
# Modes
MODE_DETECTING = 0
MODE_CALIBRATING = 1
MODE_CALIBRATING_RECORDING = 2  # New mode for active recording
MODE_CONTROLLING = 3
MODE_SIMULATED = 4  # New mode for testing without camera
MODE_ADHD_ASSISTANT = 5  # New mode for ADHD attention assistant

# Calibration
CALIBRATION_POINTS = [
    (0.1, 0.1), (0.9, 0.1),  # Top-left, Top-right
    (0.1, 0.9), (0.9, 0.9),  # Bottom-left, Bottom-right
    (0.5, 0.5)              # Center
]
CALIBRATION_SAMPLES = 15  # Increased for better accuracy
CALIBRATION_RECORD_TIME = 4  # Seconds to record samples

# Blink Detection
WINK_FRAMES_THRESHOLD = 1  # Number of frames to consider a wink (one eye closed)
BLINK_COOLDOWN_SEC = 0.5  # Prevent multiple clicks from one blink

# Gaze Smoothing
GAZE_SMOOTHING_BUFFER_SIZE = 15  # Increased from 5 to reduce shakiness
CURSOR_SMOOTHING_FACTOR = 0.3  # Lower values = more smoothing (0.1-0.9)
GAZE_DEADZONE = 1.0  # Ignore tiny movements (pixel threshold)

# Virtual Cursor
CURSOR_RADIUS = 10
CURSOR_COLOR = (255, 0, 255)  # Magenta cursor
CLICK_DURATION = 0.5  # How long to show click indication (in seconds)

# ADHD Attention Assistant Constants
ATTENTION_COOLDOWN_SEC = 5.0  # Increased from 3.0 - longer time between alerts
BORDER_THRESHOLD_PERCENT = 0.15  # Consider near border if within this % of frame edge
ATTENTION_LOST_FRAMES = 20  # Number of frames to wait before alerting
ATTENTION_ALERT_DURATION = 3.0  # How long to show the flashing alert (seconds)

# --- Pupil Detection Function ---
def find_pupil(eye_roi_gray):
    """Finds the pupil center in a grayscale eye ROI."""
    pupil_center = None
    pupil_contour = None
    
    # Ensure input is not empty
    if eye_roi_gray is None or eye_roi_gray.size == 0:
        return None, None

    try:
        # Apply image processing to enhance pupil visibility
        # 1. Increase contrast
        eye_roi_gray = cv2.equalizeHist(eye_roi_gray)
        
        # 2. Apply Gaussian Blur to reduce noise
        eye_roi_blurred = cv2.GaussianBlur(eye_roi_gray, (7, 7), 0)
        
        # 3. Use adaptive thresholding - this works better in varying lighting conditions
        # than simple thresholding
        _, threshold = cv2.threshold(eye_roi_blurred, 25, 255, cv2.THRESH_BINARY_INV)
        
        # 4. Find contours in the thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Filter contours by area to find the pupil
        if contours:
            # Get height and width of eye ROI for relative area calculation
            h, w = eye_roi_gray.shape
            
            # Define min and max pupil size as a percentage of the eye area
            min_pupil_area = (w * h) * 0.03  # 3% of eye area
            max_pupil_area = (w * h) * 0.5   # 50% of eye area
            
            # Filter valid contours by area
            valid_contours = [c for c in contours if min_pupil_area < cv2.contourArea(c) < max_pupil_area]
            
            if valid_contours:
                # Get the largest contour (likely the pupil)
                pupil_contour = max(valid_contours, key=cv2.contourArea)
                
                # Calculate moments to find center of contour
                M = cv2.moments(pupil_contour)
                if M["m00"] != 0:
                    # Get center coordinates
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    pupil_center = (cX, cY)
        
    except cv2.error as e:
        # Handle OpenCV errors gracefully
        pass
        
    return pupil_center, pupil_contour

# --- Gaze to Window Mapping Function ---
def map_gaze_to_window(gaze_offset, calibration_data, window_width, window_height):
    """Maps relative gaze offset to window coordinates based on calibration."""
    # Simple linear interpolation/scaling based on calibration extrema
    if not calibration_data or len(calibration_data) < 4:
        # Return center of window if not calibrated
        return window_width // 2, window_height // 2

    offset_x, offset_y = gaze_offset

    # Clamp offset to calibrated range to avoid going off-window drastically
    offset_x = max(calibration_data['min_x'], min(calibration_data['max_x'], offset_x))
    offset_y = max(calibration_data['min_y'], min(calibration_data['max_y'], offset_y))

    # Interpolate X
    cal_x_range = calibration_data['max_x'] - calibration_data['min_x']
    if cal_x_range == 0:  # Avoid division by zero
        window_x = window_width / 2
    else:
        window_x = ((offset_x - calibration_data['min_x']) / cal_x_range) * window_width

    # Interpolate Y
    cal_y_range = calibration_data['max_y'] - calibration_data['min_y']
    if cal_y_range == 0:  # Avoid division by zero
        window_y = window_height / 2
    else:
        window_y = ((offset_y - calibration_data['min_y']) / cal_y_range) * window_height

    # Ensure values stay within the window
    window_x = max(CURSOR_RADIUS, min(window_width - CURSOR_RADIUS, window_x))
    window_y = max(CURSOR_RADIUS, min(window_height - CURSOR_RADIUS, window_y))

    return int(window_x), int(window_y)

def main():
    # --- Screen Setup ---
    screen_width, screen_height = pyautogui.size()  # Only used for reference
    print(f"Screen size: {screen_width}x{screen_height}")
    
    # Create a larger window for the application
    window_width, window_height = 1024, 768  # Adjustable window size
    
    # --------------------

    # --- Initialize Audio ---
    pygame.mixer.init()
    attention_sound_path = "attention.mp3"
    # Verify the attention sound file exists
    if not os.path.exists(attention_sound_path):
        print(f"Warning: Attention sound file not found at {attention_sound_path}")
        print("The attention alert feature will not work without this file.")
        print("Please add an audio file named 'attention.mp3' to the application directory.")
    else:
        print(f"Attention sound loaded: {attention_sound_path}")
    # ----------------------

    # --- Load Haar Cascades ---
    # Try to find the data directory automatically
    cv_data_dir = None
    potential_paths = [
        cv2.data.haarcascades,  # OpenCV's built-in location
        '/opt/homebrew/share/opencv4/haarcascades/',  # Common path for homebrew on Apple Silicon
        '/usr/local/share/opencv4/haarcascades/'  # Common path for homebrew on Intel Mac
    ]
    
    for path in potential_paths:
        # Check for a specific file existence to confirm the directory
        if path and os.path.exists(os.path.join(path, 'haarcascade_frontalface_default.xml')):
            cv_data_dir = path
            break

    if cv_data_dir is None:
        print(f"Error: Could not automatically find the OpenCV haarcascades directory.")
        print("Please ensure OpenCV is installed correctly (e.g., via pip or homebrew).")
        print(f"Looked in: {potential_paths}")
        return

    print(f"Using Haar cascades from: {cv_data_dir}")

    face_cascade_path = os.path.join(cv_data_dir, 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(cv_data_dir, 'haarcascade_eye.xml')

    # Double check existence just in case
    if not os.path.exists(face_cascade_path) or not os.path.exists(eye_cascade_path):
        print(f"Error: Cascade files not found.")
        return

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty() or eye_cascade.empty():
        print(f"Error loading cascade classifiers.")
        return
    # --------------------------

    # --- Application State ---
    current_mode = MODE_DETECTING
    calibration_step = 0
    temp_calibration_offsets = []  # Store offsets for averaging at current point
    calibration_data = {}  # Stores {screen_point_coords: avg_gaze_offset}
    gaze_map_params = {}  # Stores {'min_x': .., 'max_x': .. etc} for mapping
    recording_start_time = 0  # For calibration recording timing
    last_frame_time = time.time()  # For FPS calculation
    
    # Wink Detection
    left_eye_closed_frames = 0
    right_eye_closed_frames = 0
    both_eyes_previously_open = True
    last_blink_time = 0
    
    # Virtual Cursor State
    cursor_pos = (window_width // 2, window_height // 2)  # Start in center
    previous_cursor_pos = cursor_pos  # For additional smoothing
    left_click_active = False
    right_click_active = False
    left_click_time = 0
    right_click_time = 0
    
    # Gaze Smoothing
    gaze_offset_buffer = deque(maxlen=GAZE_SMOOTHING_BUFFER_SIZE)
    
    # ADHD Attention Assistant State
    attention_lost_counter = 0
    last_attention_alert_time = 0
    attention_alert_active = False
    attention_alert_start_time = 0
    current_attention_message = ""
    # -------------------------

    # --- Initialize webcam or create simulated input ---
    camera_access_success = False
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("")
            print("ERROR: Could not open webcam. Check camera permissions in System Settings.")
            print("On macOS: System Settings > Privacy & Security > Camera")
            print("Make sure your terminal or IDE has permission to access the camera.")
            print("")
            print("Starting in SIMULATED mode for testing...")
            print("This will use a simulated webcam feed for testing the interface.")
            current_mode = MODE_SIMULATED
            # Create a simulated frame
            frame_width = 640
            frame_height = 480
        else:
            print("Camera opened successfully!")
            camera_access_success = True
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        print("Please check camera permissions in System Settings.")
        print("Starting in SIMULATED mode for testing...")
        current_mode = MODE_SIMULATED
        # Create a simulated frame
        frame_width = 640
        frame_height = 480
    
    # For testing/demos, use simulated mode by default during development
    if not camera_access_success:
        print("For this hackathon demo, we'll use SIMULATED mode.")
        print("In this mode, use your mouse to control the simulated eye movement.")
        print("Move your mouse around the screen to simulate eye tracking.")
        print("Move your mouse off screen or to edges to trigger attention alerts.")
        current_mode = MODE_SIMULATED
        # Create a simulated frame
        frame_width = 640
        frame_height = 480

    print("--- Controls ---")
    print(" C: Start Calibration")
    print(" S: Start Virtual Cursor Control (after calibration)")
    print(" D: Stop Control / Return to Detection Mode")
    print(" A: Start ADHD Attention Assistant Mode")
    print(" SPACE: Confirm gaze during calibration")
    print(" Q: Quit")
    print("----------------")
    print("Wink with LEFT eye for LEFT click")
    print("Wink with RIGHT eye for RIGHT click")
    print("----------------")
    print("ADHD ATTENTION ASSISTANT:")
    print("  This mode helps maintain focus by monitoring your attention.")
    print("  When you look away, move too close to edges, or leave view,")
    print("  it will play a sound alert and display a reminder message.")
    print("  Perfect for studying or work sessions requiring sustained focus.")
    print("  Press 'A' to start this mode.")
    print("----------------")
    print(f"Current Mode: DETECTING")

    while True:
        # Timing for FPS calculation
        frame_start_time = time.time()
        
        # Create a blank canvas for our window
        display_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # Read frame from webcam or create simulated frame
        if current_mode == MODE_SIMULATED:
            # Create a simulated frame
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            # Draw a face in the center
            face_x, face_y = frame_width // 2 - 50, frame_height // 2 - 50
            face_size = 100
            
            # Get mouse position for simulation
            mouse_x, mouse_y = pyautogui.position()
            screen_w, screen_h = pyautogui.size()
            
            # Check if mouse is near screen edge or off screen to simulate attention loss
            mouse_near_edge = (mouse_x < 50 or mouse_x > screen_w - 50 or 
                              mouse_y < 50 or mouse_y > screen_h - 50)
            
            # Calculate relative position of mouse in frame
            rel_mouse_x = (mouse_x / screen_w) * frame_width
            rel_mouse_y = (mouse_y / screen_h) * frame_height
            
            # For ADHD assistant testing: If mouse is near edge, move simulated face to edge
            if mouse_near_edge and current_mode == MODE_ADHD_ASSISTANT:
                # Move face toward the edge where mouse is
                if mouse_x < 50:  # Left edge
                    face_x = 0
                elif mouse_x > screen_w - 50:  # Right edge
                    face_x = frame_width - face_size
                
                if mouse_y < 50:  # Top edge
                    face_y = 0
                elif mouse_y > screen_h - 50:  # Bottom edge
                    face_y = frame_height - face_size
            
            # Draw face circle (if near edge, make it fainter to simulate partial detection)
            face_opacity = 100 if mouse_near_edge else 200
            cv2.circle(frame, (face_x + face_size//2, face_y + face_size//2), 
                      face_size//2, (face_opacity, face_opacity, face_opacity), -1)
            
            # Draw eyes 
            eye_y = face_y + face_size//3
            left_eye_x = face_x + face_size//4
            right_eye_x = face_x + 3*face_size//4
            
            # Eyes (white part)
            cv2.circle(frame, (left_eye_x, eye_y), 10, (255, 255, 255), -1)
            cv2.circle(frame, (right_eye_x, eye_y), 10, (255, 255, 255), -1)
            
            # Pupils (move with mouse for testing)
            # Calculate pupil offset based on mouse position
            pupil_x_offset = (mouse_x / screen_w - 0.5) * 5
            pupil_y_offset = (mouse_y / screen_h - 0.5) * 5
            
            # Draw pupils
            left_pupil_x = int(left_eye_x + pupil_x_offset)
            left_pupil_y = int(eye_y + pupil_y_offset)
            right_pupil_x = int(right_eye_x + pupil_x_offset) 
            right_pupil_y = int(eye_y + pupil_y_offset)
            
            # For ADHD testing, if mouse is out of screen, don't show pupils
            if mouse_near_edge and current_mode == MODE_ADHD_ASSISTANT:
                # Make pupils invisible or barely visible when simulating attention loss
                pupil_color = (100, 100, 100)  # Grey/faint
            else:
                pupil_color = (0, 0, 0)  # Black/normal
                
            cv2.circle(frame, (left_pupil_x, left_pupil_y), 4, pupil_color, -1)
            cv2.circle(frame, (right_pupil_x, right_pupil_y), 4, pupil_color, -1)
            
            # For calibration testing, simulate the pupil detection and gaze offset
            detected_pupils = [
                {
                    'pupil_abs': (left_pupil_x, left_pupil_y),
                    'eye_center_abs': (left_eye_x, eye_y),
                    'offset': (left_pupil_x - left_eye_x, left_pupil_y - eye_y),
                    'is_left': True
                },
                {
                    'pupil_abs': (right_pupil_x, right_pupil_y),
                    'eye_center_abs': (right_eye_x, eye_y),
                    'offset': (right_pupil_x - right_eye_x, right_pupil_y - eye_y),
                    'is_left': False
                }
            ]
            # Set eye detection state based on mouse position
            # If mouse is at edge or outside, simulate not detecting eyes
            left_eye_detected = not (mouse_near_edge and current_mode == MODE_ADHD_ASSISTANT)
            right_eye_detected = not (mouse_near_edge and current_mode == MODE_ADHD_ASSISTANT)
            
            ret = True
        else:
            # Use actual webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Calculate the webcam frame size for embedding in our display
        webcam_height, webcam_width = frame.shape[:2]
        
        # Resize webcam frame to fit in the corner without covering important text
        max_webcam_width = window_width // 4  # Smaller size (1/4 of window width)
        max_webcam_height = window_height // 4  # Smaller size (1/4 of window height)
        
        # Calculate scale factor to maintain aspect ratio
        scale_factor = min(max_webcam_width / webcam_width, max_webcam_height / webcam_height)
        
        # Calculate new dimensions
        new_width = int(webcam_width * scale_factor)
        new_height = int(webcam_height * scale_factor)
        
        # Resize webcam frame
        webcam_resized = cv2.resize(frame, (new_width, new_height))
        
        # Create a copy of the webcam frame for visualization
        webcam_visual = webcam_resized.copy()
        
        # Position in bottom-left corner with margin to avoid calibration dots
        webcam_x = 10
        webcam_y = window_height - new_height - 10

        # Create background for webcam area
        cv2.rectangle(display_frame, 
                     (webcam_x-1, webcam_y-1), 
                     (webcam_x+new_width+1, webcam_y+new_height+1), 
                     (40, 40, 40), -1)
        
        # Draw border around the webcam feed
        cv2.rectangle(display_frame, 
                     (webcam_x-1, webcam_y-1), 
                     (webcam_x+new_width+1, webcam_y+new_height+1), 
                     (255, 255, 255), 1)
        
        # Add title above webcam
        cv2.putText(display_frame, "Eye Tracking Feed", 
                   (webcam_x, webcam_y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Convert frame to grayscale for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Face Detection ---
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90)
        )

        # Initialize variables for current frame
        detected_pupils = []  # Store pupil data for this frame
        left_eye_detected = False
        right_eye_detected = False
        face_center = None

        if len(faces) > 0:
            # Use the largest face (assuming it's the user)
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_center = (x + w // 2, y + h // 2)
            
            # Draw rectangle around face (on the webcam image for reference)
            cv2.rectangle(webcam_visual, 
                         (int(x * scale_factor), int(y * scale_factor)), 
                         (int((x + w) * scale_factor), int((y + h) * scale_factor)), 
                         (255, 0, 0), 2)
            
            # Extract face region of interest (ROI)
            roi_gray = gray_frame[y:y + h, x:x + w]
            
            # --- Eye Detection ---
            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.05, minNeighbors=8,
                minSize=(w // 10, h // 10), maxSize=(w // 3, h // 3)
            )
            
            # Identify left/right eye based on x-position
            # (left eye is on the right side of the image due to flipping)
            eyes_sorted = sorted(eyes, key=lambda e: e[0])  # Sort by x-coord
            processed_eyes = 0
            
            for i, (ex, ey, ew, eh) in enumerate(eyes_sorted):
                # Basic check: ensure eye is in upper part of face
                if ey + eh // 2 > h * 0.6 or processed_eyes >= 2:
                    continue
                
                # Is this eye on the left or right side of the face?
                is_left_eye = (ex + ew // 2) < w // 2
                
                # Calculate absolute coordinates
                eye_x_abs, eye_y_abs = x + ex, y + ey
                eye_w, eye_h = ew, eh
                
                # Draw rectangle around the eye (on the webcam image)
                scaled_eye_x = int(eye_x_abs * scale_factor)
                scaled_eye_y = int(eye_y_abs * scale_factor)
                scaled_eye_w = int(eye_w * scale_factor)
                scaled_eye_h = int(eye_h * scale_factor)
                
                # Draw eye status label
                eye_label = "LEFT" if is_left_eye else "RIGHT"
                label_pos = (scaled_eye_x, scaled_eye_y - 5)
                cv2.putText(webcam_visual, eye_label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                cv2.rectangle(webcam_visual, 
                             (scaled_eye_x, scaled_eye_y), 
                             (scaled_eye_x + scaled_eye_w, scaled_eye_y + scaled_eye_h), 
                             (0, 255, 0), 1)
                
                # Extract eye region
                eye_roi_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                
                # Track which eye was detected
                if is_left_eye:
                    left_eye_detected = True
                else:
                    right_eye_detected = True
                
                # Detect pupil
                pupil_center_rel, pupil_contour = find_pupil(eye_roi_gray)
                
                if pupil_center_rel:
                    # Calculate pupil's absolute position
                    pupil_x_rel, pupil_y_rel = pupil_center_rel
                    pupil_x_abs = eye_x_abs + pupil_x_rel
                    pupil_y_abs = eye_y_abs + pupil_y_rel
                    
                    # Draw pupil center (on the webcam image)
                    scaled_pupil_x = int(pupil_x_abs * scale_factor)
                    scaled_pupil_y = int(pupil_y_abs * scale_factor)
                    
                    # Draw pupil with more visibility
                    cv2.circle(webcam_visual, (scaled_pupil_x, scaled_pupil_y), 3, (0, 0, 255), -1)
                    cv2.circle(webcam_visual, (scaled_pupil_x, scaled_pupil_y), 5, (0, 0, 255), 1)
                    
                    # Display "PUPIL DETECTED" near the eye
                    pupil_text_pos = (scaled_eye_x, scaled_eye_y + scaled_eye_h + 15)
                    cv2.putText(webcam_visual, "PUPIL OK", pupil_text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # If we have the pupil contour, draw it on the eye image
                    if pupil_contour is not None:
                        try:
                            # Scale contour to display size
                            scaled_contour = pupil_contour.copy() * scale_factor
                            scaled_contour = scaled_contour.astype(np.int32)
                            
                            # Shift contour to the eye's position
                            shifted_contour = scaled_contour + np.array([scaled_eye_x, scaled_eye_y])
                            
                            # Draw the contour
                            cv2.drawContours(webcam_visual, [shifted_contour], -1, (0, 255, 255), 1)
                        except Exception as e:
                            # Skip drawing contour if there's an error
                            pass
                    
                    # Calculate eye center
                    eye_center_x_abs = eye_x_abs + eye_w // 2
                    eye_center_y_abs = eye_y_abs + eye_h // 2
                    
                    # Store pupil data
                    detected_pupils.append({
                        'pupil_abs': (pupil_x_abs, pupil_y_abs),
                        'eye_center_abs': (eye_center_x_abs, eye_center_y_abs),
                        'offset': (pupil_x_abs - eye_center_x_abs, pupil_y_abs - eye_center_y_abs),
                        'is_left': is_left_eye
                    })
                else:
                    # Display "NO PUPIL" message
                    pupil_text_pos = (scaled_eye_x, scaled_eye_y + scaled_eye_h + 15)
                    cv2.putText(webcam_visual, "NO PUPIL", pupil_text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                processed_eyes += 1

        # --- Improved Wink Detection Logic ---
        # Track if both eyes are currently visible
        both_eyes_open = left_eye_detected and right_eye_detected
        
        # Update wink detection counters
        if both_eyes_previously_open:
            if left_eye_detected and not right_eye_detected:
                # Left eye open, right eye closed (right eye wink)
                right_eye_closed_frames += 1
                left_eye_closed_frames = 0
            elif right_eye_detected and not left_eye_detected:
                # Right eye open, left eye closed (left eye wink)
                left_eye_closed_frames += 1
                right_eye_closed_frames = 0
            elif both_eyes_open:
                # Both eyes still open
                left_eye_closed_frames = 0
                right_eye_closed_frames = 0
            else:
                # Both eyes closed (normal blink)
                left_eye_closed_frames = 0
                right_eye_closed_frames = 0
        
        # Save current eye state for next frame
        both_eyes_previously_open = both_eyes_open
        
        # --- Calculate Average Gaze Offset ---
        avg_gaze_offset = None
        if len(detected_pupils) > 0:
            avg_offset_x = sum(p['offset'][0] for p in detected_pupils) / len(detected_pupils)
            avg_offset_y = sum(p['offset'][1] for p in detected_pupils) / len(detected_pupils)
            avg_gaze_offset = (avg_offset_x, avg_offset_y)
            gaze_offset_buffer.append(avg_gaze_offset)
        elif current_mode == MODE_CONTROLLING and len(gaze_offset_buffer) > 0:
            # If eyes lost, use the last known good offset for smoothing
            avg_gaze_offset = gaze_offset_buffer[-1]
        
        # --- Apply Enhanced Smoothing ---
        smoothed_gaze_offset = None
        if len(gaze_offset_buffer) > 0:
            # Advanced smoothing with exponential moving average
            # This gives more weight to recent measurements while retaining some history
            weights = np.exp(np.linspace(-1, 0, len(gaze_offset_buffer)))
            weights /= weights.sum() # Normalize weights
            
            # Apply weights to x and y coordinates
            smooth_x = sum(g[0] * w for g, w in zip(gaze_offset_buffer, weights))
            smooth_y = sum(g[1] * w for g, w in zip(gaze_offset_buffer, weights))
            smoothed_gaze_offset = (smooth_x, smooth_y)

        # --- ADHD Attention Assistant Logic ---
        if current_mode == MODE_ADHD_ASSISTANT:
            # Display cursor in ADHD mode (similar to CONTROLLING mode)
            if smoothed_gaze_offset and gaze_map_params:
                # Map gaze offset to window coordinates for the virtual cursor
                target_cursor_pos = map_gaze_to_window(smoothed_gaze_offset, 
                                                     gaze_map_params, 
                                                     window_width, window_height)
                
                # Apply deadzone to reduce shakiness when eyes are relatively still
                dx = target_cursor_pos[0] - previous_cursor_pos[0]
                dy = target_cursor_pos[1] - previous_cursor_pos[1]
                
                # Only move cursor if change exceeds deadzone
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < GAZE_DEADZONE:
                    # Use previous position to reduce jitter
                    target_cursor_pos = previous_cursor_pos
                
                # Apply additional smoothing between frames for smoother cursor movement
                cursor_x = int(previous_cursor_pos[0] * (1-CURSOR_SMOOTHING_FACTOR) + 
                              target_cursor_pos[0] * CURSOR_SMOOTHING_FACTOR)
                cursor_y = int(previous_cursor_pos[1] * (1-CURSOR_SMOOTHING_FACTOR) + 
                              target_cursor_pos[1] * CURSOR_SMOOTHING_FACTOR)
                
                cursor_pos = (cursor_x, cursor_y)
                previous_cursor_pos = cursor_pos  # Save for next frame
            else:
                # If no valid gaze data, put cursor in center
                cursor_pos = (window_width // 2, window_height // 2)
                previous_cursor_pos = cursor_pos
            
            # Always draw cursor
            cv2.circle(display_frame, cursor_pos, CURSOR_RADIUS, CURSOR_COLOR, -1)
            
            # Check if attention is lost based on various conditions
            frame_height, frame_width = frame.shape[:2]
            border_x = int(frame_width * BORDER_THRESHOLD_PERCENT)
            border_y = int(frame_height * BORDER_THRESHOLD_PERCENT)
            
            attention_issue = False
            attention_message = ""
            
            # Check if no face detected
            if len(faces) == 0:
                attention_issue = True
                attention_message = "No face detected"
            
            # Check if no eyes detected
            elif not (left_eye_detected or right_eye_detected):
                attention_issue = True
                attention_message = "Eyes not detected"
            
            # Check if face is near border (looking away)
            elif face_center:
                face_x, face_y = face_center
                if (face_x < border_x or face_x > frame_width - border_x or 
                    face_y < border_y or face_y > frame_height - border_y):
                    attention_issue = True
                    attention_message = "Face near edge"
            
            # Update attention counter
            if attention_issue:
                attention_lost_counter += 1
                # Draw warning text
                cv2.putText(display_frame, f"Attention: {attention_message}", 
                           (window_width // 3, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            else:
                attention_lost_counter = 0
                # Draw positive feedback
                cv2.putText(display_frame, "Attention: Good focus", 
                           (window_width // 3, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
            
            # Trigger attention alert if threshold reached and cooldown expired
            now = time.time()
            
            # Check if we need to deactivate an ongoing alert
            if attention_alert_active and (now - attention_alert_start_time >= ATTENTION_ALERT_DURATION):
                attention_alert_active = False
                print("Attention alert ended.")
            
            # Start a new alert if needed
            if (not attention_alert_active and 
                attention_lost_counter >= ATTENTION_LOST_FRAMES and 
                now - last_attention_alert_time > ATTENTION_COOLDOWN_SEC):
                # Play attention sound if file exists
                if os.path.exists(attention_sound_path):
                    try:
                        pygame.mixer.music.load(attention_sound_path)
                        pygame.mixer.music.play()
                        print("Attention alert triggered!")
                    except Exception as e:
                        print(f"Error playing attention sound: {e}")
                
                # Start new alert
                attention_alert_active = True
                attention_alert_start_time = now
                last_attention_alert_time = now
                current_attention_message = attention_message
            
            # Show flashing alert if active
            if attention_alert_active:
                # Calculate flashing effect parameters
                time_in_alert = now - attention_alert_start_time
                
                # Create flashing effect by varying intensity and colors based on time
                # Multiple flashing effects combined for maximum attention grabbing
                
                # 1. Determine flash intensity (0.0 to 1.0) using sine wave for smooth pulsing
                flash_speed = 10.0  # Hz - higher = faster flashing
                flash_intensity = 0.5 + 0.5 * np.sin(time_in_alert * flash_speed * 2 * np.pi)
                
                # 2. Create color alternation effect
                color_cycle_speed = 5.0  # Hz
                color_phase = (time_in_alert * color_cycle_speed) % 3.0
                
                if color_phase < 1.0:
                    # Red phase
                    flash_color = (0, 0, 255)
                elif color_phase < 2.0:
                    # Yellow phase
                    flash_color = (0, 255, 255)
                else:
                    # Blue phase
                    flash_color = (255, 0, 0)
                
                # Create flashing overlay with varying opacity
                overlay = display_frame.copy()
                
                # Fill with current flash color
                cv2.rectangle(overlay, (0, 0), (window_width, window_height), 
                            flash_color, -1)
                
                # Apply varying opacity based on flash intensity
                alpha = 0.2 + 0.3 * flash_intensity  # Opacity varies between 0.2-0.5
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                
                # Add border that pulses in thickness
                border_thickness = int(5 + 15 * flash_intensity)  # Thickness varies between 5-20px
                cv2.rectangle(display_frame, (0, 0), (window_width, window_height), 
                            flash_color, border_thickness)
                
                # 3. Create a prominent text box in the center that moves slightly
                text_box_width = window_width - 100
                text_box_height = 200
                
                # Make the text box position jitter slightly to grab attention
                jitter_amount = 10
                jitter_x = int(jitter_amount * np.sin(time_in_alert * 15))
                jitter_y = int(jitter_amount * np.cos(time_in_alert * 12))
                
                text_box_x = (window_width - text_box_width) // 2 + jitter_x
                text_box_y = (window_height - text_box_height) // 2 + jitter_y
                
                # Draw semi-transparent black background for text
                cv2.rectangle(display_frame, 
                            (text_box_x, text_box_y),
                            (text_box_x + text_box_width, text_box_y + text_box_height),
                            (0, 0, 0), -1)
                cv2.rectangle(display_frame, 
                            (text_box_x, text_box_y),
                            (text_box_x + text_box_width, text_box_y + text_box_height),
                            flash_color, 5)  # Border uses flash color
                
                # 4. Flashing text for maximum effect - alternate size and intensity
                text_size = 1.6 + 0.4 * flash_intensity  # Size varies between 1.6-2.0
                
                # Main attention text
                cv2.putText(display_frame, "GET BACK TO ATTENTION!", 
                        (window_width // 2 - 300 + jitter_x, window_height // 2 - 30 + jitter_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 5)
                
                # Add a glow/halo effect with the flash color
                cv2.putText(display_frame, "GET BACK TO ATTENTION!", 
                        (window_width // 2 - 300 + jitter_x, window_height // 2 - 30 + jitter_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, flash_color, 10)
                cv2.putText(display_frame, "GET BACK TO ATTENTION!", 
                        (window_width // 2 - 300 + jitter_x, window_height // 2 - 30 + jitter_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 2)
                
                # 5. Add the specific attention issue
                cv2.putText(display_frame, f"Issue: {current_attention_message}", 
                        (window_width // 2 - 150 + jitter_x, window_height // 2 + 50 + jitter_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, flash_color, 3)
                cv2.putText(display_frame, f"Issue: {current_attention_message}", 
                        (window_width // 2 - 150 + jitter_x, window_height // 2 + 50 + jitter_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
                
                # Also add a countdown to show how long until the alert ends
                time_left = ATTENTION_ALERT_DURATION - time_in_alert
                cv2.putText(display_frame, f"Alert ends in: {time_left:.1f}s", 
                        (window_width // 2 - 150 + jitter_x, window_height // 2 + 100 + jitter_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Draw attention status bar
            attention_bar_width = 200
            attention_bar_height = 20
            bar_x = window_width - attention_bar_width - 20
            bar_y = 50
            
            # Draw bar background
            cv2.rectangle(display_frame, (bar_x, bar_y), 
                         (bar_x + attention_bar_width, bar_y + attention_bar_height), 
                         (100, 100, 100), -1)
            
            # Calculate fill level (red when approaching threshold)
            fill_width = int((attention_lost_counter / ATTENTION_LOST_FRAMES) * attention_bar_width)
            fill_width = min(fill_width, attention_bar_width)
            
            # Choose color based on how close to threshold
            if attention_lost_counter < ATTENTION_LOST_FRAMES * 0.5:
                fill_color = (0, 255, 0)  # Green
            elif attention_lost_counter < ATTENTION_LOST_FRAMES * 0.8:
                fill_color = (0, 255, 255)  # Yellow
            else:
                fill_color = (0, 0, 255)  # Red
            
            # Draw filled portion
            cv2.rectangle(display_frame, (bar_x, bar_y), 
                         (bar_x + fill_width, bar_y + attention_bar_height), 
                         fill_color, -1)
            
            # Draw label
            cv2.putText(display_frame, "Attention Tracker", 
                       (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)

        # --- Mode-Specific Logic ---
        
        # == CALIBRATION MODE ==
        if current_mode == MODE_CALIBRATING:
            if calibration_step < len(CALIBRATION_POINTS):
                # Display current calibration target
                target_rel_x, target_rel_y = CALIBRATION_POINTS[calibration_step]
                target_draw_x = int(target_rel_x * window_width) 
                target_draw_y = int(target_rel_y * window_height)

                # Draw target on screen with more visibility
                cv2.circle(display_frame, (target_draw_x, target_draw_y), 25, (0, 255, 255), 2)  # Yellow circle
                cv2.circle(display_frame, (target_draw_x, target_draw_y), 8, (0, 0, 255), -1)    # Red center dot
                
                # Add crosshair for better targeting
                cv2.line(display_frame, (target_draw_x - 30, target_draw_y), (target_draw_x + 30, target_draw_y), (255, 255, 255), 1)
                cv2.line(display_frame, (target_draw_x, target_draw_y - 30), (target_draw_x, target_draw_y + 30), (255, 255, 255), 1)
                
                # Clear and prominent instructions - larger, centered text box
                instr_y = window_height // 4
                cv2.rectangle(display_frame, (50, instr_y-40), (window_width-50, instr_y+40), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (50, instr_y-40), (window_width-50, instr_y+40), (0, 100, 200), 2)
                
                # Main calibration instruction
                cv2.putText(display_frame, f"CALIBRATION STEP {calibration_step+1} OF {len(CALIBRATION_POINTS)}", 
                           (window_width//4, instr_y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                
                # Detailed instruction
                cv2.putText(display_frame, "Focus on the target dot and press SPACE to record", 
                           (window_width//5, instr_y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Add point location label
                location_labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"]
                if calibration_step < len(location_labels):
                    cv2.putText(display_frame, f"Target: {location_labels[calibration_step]}", 
                               (target_draw_x - 80, target_draw_y - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            else:
                # Calibration finished - show attractive completion message
                cv2.rectangle(display_frame, (window_width//4-20, window_height//2-60), 
                             (3*window_width//4+20, window_height//2+60), (0, 50, 0), -1)
                cv2.rectangle(display_frame, (window_width//4-20, window_height//2-60), 
                             (3*window_width//4+20, window_height//2+60), (0, 255, 0), 2)
                
                cv2.putText(display_frame, "CALIBRATION COMPLETE!", 
                           (window_width//4, window_height//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                cv2.putText(display_frame, "Press 'S' to Start Cursor Control", 
                           (window_width//4+50, window_height//2+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        elif current_mode == MODE_CALIBRATING_RECORDING:
            # Recording calibration samples
            target_rel_x, target_rel_y = CALIBRATION_POINTS[calibration_step]
            target_draw_x = int(target_rel_x * window_width) 
            target_draw_y = int(target_rel_y * window_height)
            
            # Draw target with progress indicator and larger target
            time_elapsed = frame_start_time - recording_start_time
            angle = int(360 * time_elapsed / CALIBRATION_RECORD_TIME)
            if angle > 360:
                angle = 360
            
            # Draw progress arc
            cv2.circle(display_frame, (target_draw_x, target_draw_y), 40, (0, 100, 255), 2)  # Larger outer circle
            cv2.ellipse(display_frame, (target_draw_x, target_draw_y), (40, 40), 
                       0, 0, angle, (0, 255, 0), 3)  # Progress arc
            
            # Draw target - make it more prominent
            cv2.circle(display_frame, (target_draw_x, target_draw_y), 25, (0, 255, 255), 2)  # Yellow circle
            cv2.circle(display_frame, (target_draw_x, target_draw_y), 8, (0, 0, 255), -1)    # Red center dot
            
            # Draw crosshair
            cv2.line(display_frame, (target_draw_x - 30, target_draw_y), (target_draw_x + 30, target_draw_y), (255, 255, 255), 1)
            cv2.line(display_frame, (target_draw_x, target_draw_y - 30), (target_draw_x, target_draw_y + 30), (255, 255, 255), 1)
            
            # Clear and prominent instructions
            instr_y = 50
            cv2.rectangle(display_frame, (0, instr_y-30), (window_width, instr_y+10), (0, 0, 0), -1)
            cv2.putText(display_frame, f"RECORDING POINT {calibration_step+1}: Keep looking at the target! ({len(temp_calibration_offsets)}/{CALIBRATION_SAMPLES} samples)",
                       (window_width//8, instr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show countdown
            time_left = CALIBRATION_RECORD_TIME - time_elapsed
            if time_left > 0:
                cv2.putText(display_frame, f"Time left: {time_left:.1f}s", 
                           (target_draw_x - 70, target_draw_y - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Collect data only during recording mode
            if avg_gaze_offset is not None:
                temp_calibration_offsets.append(avg_gaze_offset)
                
            # Check if we have enough samples or time is up
            if len(temp_calibration_offsets) >= CALIBRATION_SAMPLES or time_elapsed >= CALIBRATION_RECORD_TIME:
                # Average the collected offsets for this point
                if temp_calibration_offsets:
                    # Use median filtering to remove outliers
                    all_x = [o[0] for o in temp_calibration_offsets]
                    all_y = [o[1] for o in temp_calibration_offsets]
                    all_x.sort()
                    all_y.sort()
                    
                    # Remove outliers (20% from each end)
                    trim_count = len(all_x) // 5
                    if trim_count > 0:
                        all_x = all_x[trim_count:-trim_count]
                        all_y = all_y[trim_count:-trim_count]
                    
                    # Get average of remaining values
                    avg_x = sum(all_x) / len(all_x)
                    avg_y = sum(all_y) / len(all_y)

                    # Store the calibration data point
                    target_coords = CALIBRATION_POINTS[calibration_step]
                    calibration_data[target_coords] = (avg_x, avg_y)
                    print(f"  Calibrated point {calibration_step + 1}: Screen={target_coords}, Gaze Offset=({avg_x:.2f}, {avg_y:.2f})")

                # Move to next step
                calibration_step += 1
                temp_calibration_offsets = []  # Reset for next point
                
                # Switch back to normal calibration mode
                current_mode = MODE_CALIBRATING
                
                # If calibration just finished, calculate mapping params
                if calibration_step >= len(CALIBRATION_POINTS):
                    print("Calculating gaze map parameters...")
                    all_offsets = list(calibration_data.values())
                    if len(all_offsets) >= 2:  # Need at least 2 points
                        gaze_map_params['min_x'] = min(o[0] for o in all_offsets)
                        gaze_map_params['max_x'] = max(o[0] for o in all_offsets)
                        gaze_map_params['min_y'] = min(o[1] for o in all_offsets)
                        gaze_map_params['max_y'] = max(o[1] for o in all_offsets)
                        print(f"  Gaze Map Params: {gaze_map_params}")
                    else:
                        print("  Error: Not enough calibration points to create map.")
                        gaze_map_params = {}  # Reset if failed
                else:
                    # Short pause between calibration points
                    time.sleep(0.5)

        # == CURSOR CONTROL MODE ==
        elif current_mode == MODE_CONTROLLING:
            if smoothed_gaze_offset and gaze_map_params:
                # Map gaze offset to window coordinates for the virtual cursor
                target_cursor_pos = map_gaze_to_window(smoothed_gaze_offset, 
                                                     gaze_map_params, 
                                                     window_width, window_height)
                
                # Apply deadzone to reduce shakiness when eyes are relatively still
                dx = target_cursor_pos[0] - previous_cursor_pos[0]
                dy = target_cursor_pos[1] - previous_cursor_pos[1]
                
                # Only move cursor if change exceeds deadzone
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < GAZE_DEADZONE:
                    # Use previous position to reduce jitter
                    target_cursor_pos = previous_cursor_pos
                
                # Apply additional smoothing between frames for smoother cursor movement
                cursor_x = int(previous_cursor_pos[0] * (1-CURSOR_SMOOTHING_FACTOR) + 
                              target_cursor_pos[0] * CURSOR_SMOOTHING_FACTOR)
                cursor_y = int(previous_cursor_pos[1] * (1-CURSOR_SMOOTHING_FACTOR) + 
                              target_cursor_pos[1] * CURSOR_SMOOTHING_FACTOR)
                
                cursor_pos = (cursor_x, cursor_y)
                previous_cursor_pos = cursor_pos  # Save for next frame
                
                # --- Process Clicks from Winks ---
                can_click = (frame_start_time - last_blink_time) > BLINK_COOLDOWN_SEC
                
                if can_click:
                    if left_eye_closed_frames >= WINK_FRAMES_THRESHOLD:
                        print("Left Click!")
                        left_click_active = True
                        left_click_time = frame_start_time
                        last_blink_time = frame_start_time
                        left_eye_closed_frames = 0
                    
                    if right_eye_closed_frames >= WINK_FRAMES_THRESHOLD:
                        print("Right Click!")
                        right_click_active = True
                        right_click_time = frame_start_time
                        last_blink_time = frame_start_time
                        right_eye_closed_frames = 0
            else:
                # If no valid gaze data, put cursor in center
                cursor_pos = (window_width // 2, window_height // 2)
                previous_cursor_pos = cursor_pos  # Save for next frame
                cv2.putText(display_frame, "WARN: Not calibrated or no gaze detected",
                           (window_width // 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw virtual cursor
            # Check if click feedback should be shown
            if left_click_active and frame_start_time - left_click_time < CLICK_DURATION:
                # Draw left click feedback (red circle around cursor)
                cv2.circle(display_frame, cursor_pos, CURSOR_RADIUS + 10, (0, 0, 255), 4)
                # Show left click text
                cv2.putText(display_frame, "LEFT CLICK", 
                           (cursor_pos[0] - 50, cursor_pos[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif right_click_active and frame_start_time - right_click_time < CLICK_DURATION:
                # Draw right click feedback (blue circle around cursor)
                cv2.circle(display_frame, cursor_pos, CURSOR_RADIUS + 10, (255, 0, 0), 4)
                # Show right click text
                cv2.putText(display_frame, "RIGHT CLICK", 
                           (cursor_pos[0] - 50, cursor_pos[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Clear click feedback after duration
            if left_click_active and frame_start_time - left_click_time >= CLICK_DURATION:
                left_click_active = False
            if right_click_active and frame_start_time - right_click_time >= CLICK_DURATION:
                right_click_active = False
                
            # Always draw cursor
            cv2.circle(display_frame, cursor_pos, CURSOR_RADIUS, CURSOR_COLOR, -1)
                
            # Show wink stats (for debugging)
            cv2.putText(display_frame, f"Left eye closed frames: {left_eye_closed_frames}", 
                       (10, window_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Right eye closed frames: {right_eye_closed_frames}", 
                       (10, window_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # == DEFAULT DETECTING MODE ==
        else:  # MODE_DETECTING
            # Show basic information for debugging
            if avg_gaze_offset:
                cv2.putText(display_frame, f"Gaze offset: ({avg_gaze_offset[0]:.1f}, {avg_gaze_offset[1]:.1f})", 
                           (window_width // 3, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show eye status
            status_text = f"Eyes: {'Both' if both_eyes_open else ('Left' if left_eye_detected else ('Right' if right_eye_detected else 'None'))}"
            cv2.putText(display_frame, status_text, 
                       (window_width // 3, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display Mode Info
        if current_mode == MODE_SIMULATED:
            mode_text = "Mode: SIMULATION (Move mouse to control eyes)"
            cv2.putText(display_frame, mode_text, (window_width // 3 - 100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "Camera access denied - using simulated input", 
                       (window_width // 3 - 100, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        elif current_mode == MODE_ADHD_ASSISTANT:
            mode_text = "Mode: ADHD ATTENTION ASSISTANT"
            cv2.putText(display_frame, mode_text, (window_width // 3, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            mode_text = f"Mode: {'DETECTING' if current_mode == MODE_DETECTING else ('CALIBRATING' if current_mode == MODE_CALIBRATING else ('RECORDING' if current_mode == MODE_CALIBRATING_RECORDING else 'CONTROLLING'))}"
            cv2.putText(display_frame, mode_text, (window_width // 3, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show smoothing information
        cv2.putText(display_frame, f"Smoothing: {GAZE_SMOOTHING_BUFFER_SIZE} frames, Deadzone: {GAZE_DEADZONE}px", 
                   (10, window_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Update the webcam with pupil detection information in the display frame
        display_frame[webcam_y:webcam_y+new_height, webcam_x:webcam_x+new_width] = webcam_visual

        # Display the resulting frame
        cv2.imshow('Eye Tracking', display_frame)

        # --- Handle Key Presses ---
        key = cv2.waitKey(1) & 0xFF

        # Check if attention alert is active and any key pressed to dismiss it
        if attention_alert_active and key != 255:  # 255 means no key pressed
            attention_alert_active = False
            print("Attention alert dismissed by user.")

        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('c'):
            if current_mode != MODE_CALIBRATING and current_mode != MODE_CALIBRATING_RECORDING:
                print("Starting Calibration...")
                current_mode = MODE_CALIBRATING
                calibration_step = 0
                temp_calibration_offsets = []
                calibration_data = {}  # Clear previous calibration
                gaze_map_params = {}
                gaze_offset_buffer.clear()
        elif key == ord('s'):
            if calibration_step >= len(CALIBRATION_POINTS):
                if gaze_map_params:
                    print("Starting Cursor Control...")
                    current_mode = MODE_CONTROLLING
                    gaze_offset_buffer.clear()  # Clear buffer before starting control
                else:
                    print("Cannot start control: Calibration incomplete or failed.")
            elif current_mode == MODE_DETECTING:
                print("Please complete calibration ('c') before starting control ('s').")

        elif key == ord('d'):
            if current_mode != MODE_DETECTING:
                print("Stopping and returning to Detection Mode.")
                current_mode = MODE_DETECTING

        elif key == ord('a'):  # New key for ADHD Assistant mode
            print("Starting ADHD Attention Assistant...")
            current_mode = MODE_ADHD_ASSISTANT
            attention_lost_counter = 0
            last_attention_alert_time = 0
            
            # If not calibrated yet, create a simple default calibration
            # This allows the cursor to work without going through calibration
            if not gaze_map_params:
                print("  No calibration found. Using default values for cursor control.")
                # Create simple default mapping parameters based on typical ranges
                # These are rough estimates that will work for basic cursor movement
                gaze_map_params = {
                    'min_x': -30.0,  # Typical minimum gaze offset X
                    'max_x': 30.0,   # Typical maximum gaze offset X
                    'min_y': -20.0,  # Typical minimum gaze offset Y
                    'max_y': 20.0    # Typical maximum gaze offset Y
                }
                # Clear any existing buffer for fresh start
                gaze_offset_buffer.clear()

        elif key == ord(' '):  # Spacebar
            if current_mode == MODE_CALIBRATING and calibration_step < len(CALIBRATION_POINTS):
                # Start recording for this calibration point
                current_mode = MODE_CALIBRATING_RECORDING
                recording_start_time = frame_start_time
                temp_calibration_offsets = []  # Clear any previous samples
                print(f"  Started recording for calibration point {calibration_step + 1}")

        # Update FPS calculation
        frame_duration = time.time() - frame_start_time
        fps = 1.0 / max(frame_duration, 0.001)  # Avoid division by zero
        last_frame_time = frame_start_time

        # Add performance metrics to display
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Release webcam and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()  # Clean up pygame
    print("Webcam feed stopped.")

if __name__ == "__main__":
    main() 
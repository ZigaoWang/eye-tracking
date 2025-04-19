# ADHD Attention Assistant

## Project Overview
This project was developed for the hackathon theme "科技无障碍 AI" (Technology Accessibility AI). It uses computer vision and AI to help ADHD individuals maintain focus during work or study sessions.

## Key Features

### ADHD Attention Assistant Mode
- **Purpose**: Helps users stay focused by monitoring their attention via webcam
- **How it works**: 
  - Detects when the user is looking away from the screen
  - Detects when the user's face/eyes aren't visible to the camera
  - Provides immediate visual and audio alerts when attention is lost
  - Shows an eye-catching flashing alert that demands attention
  - Tracks focus metrics in a status bar
  - **NEW: Focus Reward System** - Plays a celebratory sound and displays positive feedback when user maintains focus for 10 consecutive seconds
  - **NEW: Brief Distraction Tolerance** - Allows short distractions under 2 seconds without resetting focus (for picking up a pen, scratching, etc.)
  - **NEW: Session Statistics** - Tracks total session time, focus time, and focus percentage
  - Visually displays focus progress with a progress bar
  - Enhanced reward screen with congratulatory message, confetti effects, and countdown timer

### Additional Features
- Cursor control via eye tracking (requires calibration)
- Wink detection for click actions
- Simulated mode for testing without camera access

## Requirements
- Python 3.7+
- OpenCV (cv2)
- NumPy
- PyAutoGUI
- Pygame (for audio alerts)

## Installation
```bash
pip install opencv-python numpy pyautogui pygame
```

## Audio Files
The application requires two audio files:
- `attention.mp3` - Played when attention is lost
- `wow.mp3` - Played as a reward when focus is maintained for 10 seconds

Place these in the same directory as the application.

## Usage

1. Run the application:
```bash
python main.py
```

2. You'll see a webcam feed and detection information. Press keys to control the application:
   - `A`: Start ADHD Attention Assistant mode
   - `T`: Toggle session statistics display
   - `C`: Start calibration (for cursor control)
   - `S`: Start cursor control (after calibration)
   - `D`: Return to detection mode
   - `Q`: Quit

### Using ADHD Attention Assistant Mode
1. Press `A` to activate
2. Position yourself in front of the camera
3. Continue your work as normal
4. Brief distractions (under 2 seconds) are tolerated without resetting focus
5. If you look away or get distracted for longer periods, the system will flash an alert and play a sound
6. If you maintain focus for 10 seconds, you'll receive a positive reward sound and visual feedback
7. Press `T` to view session statistics (total time, focus time, focus percentage)
8. Press any key to dismiss an active alert or reward screen

### Camera Permissions
On macOS, you'll need to grant camera permissions:
1. Go to System Settings > Privacy & Security > Camera
2. Allow your terminal or IDE to access the camera

### Simulation Mode
If camera access isn't available, the application will run in simulation mode:
- Move your mouse around the screen to control the simulated eye movement
- Move your mouse to the edge of the screen to trigger attention alerts
- Keep your mouse in the center of the screen to trigger focus rewards

## Hackathon Context
This project addresses accessibility needs for individuals with ADHD by providing technology that:
1. Reduces distractions
2. Helps maintain focus during important tasks
3. Creates awareness of attention patterns
4. Provides immediate feedback when focus shifts
5. Offers positive reinforcement for sustained attention
6. Understands natural behavior by allowing brief distractions
7. Provides quantitative feedback on focus performance

The AI component uses computer vision to detect face position, eye states, and gaze direction, then applies algorithms to determine attention state and provide appropriate interventions. 
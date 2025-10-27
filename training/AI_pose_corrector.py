import cv2
import mediapipe as mp
import numpy as np
import json
import tensorflow as tf
from collections import deque
import time
import os

# --- Import the shared logic ---
from analysis_logic import ExerciseAnalyzer

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use the 'training' folder for consistency
TRAINING_ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, 'training')
MODEL_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'exercise_classifier_lstm.h5')
LABEL_MAPPING_FILENAME = os.path.join(TRAINING_ARTIFACTS_DIR, 'label_mapping.json')

SEQUENCE_LENGTH = 90
CONF_THRESHOLD = 0.80 
STABILITY_FRAMES = 10
UI_COLOR = (0, 150, 255)
PREDICTION_INTERVAL = 3

# --- MAIN EXECUTION ---

def main():
    print("Loading model and labels...")
    try:
        model = tf.keras.models.load_model(MODEL_FILENAME)
        with open(LABEL_MAPPING_FILENAME, 'r') as f:
            label_mapping = json.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure '{MODEL_FILENAME}' and '{LABEL_MAPPING_FILENAME}' exist.")
        return

    print("Initializing MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # --- Initialize Analyzer ---
    analyzer = ExerciseAnalyzer(
        sequence_length=SEQUENCE_LENGTH,
        conf_threshold=CONF_THRESHOLD,
        stability_frames=STABILITY_FRAMES
    )
    
    # TODO: This should be dynamic, but we'll hardcode for testing
    current_exercise = "bicep_curl" 

    print("Starting webcam feed...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Performance boost
        
        # Process the image and find pose
        results = pose.process(image_rgb)
        
        image_rgb.flags.writeable = True
        
        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
            image_rgb,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # --- AI Analysis ---
        rep_counter = analyzer.rep_counter
        form_status = analyzer.form_status
        stable_prediction = analyzer.stable_prediction
        debug_angles = {}
        
        if results.pose_landmarks:
            # The analyzer handles everything:
            # feature extraction, prediction, rep counting, and form logic
            # This is the ONLY call needed.
            rep_counter, form_status, stable_prediction, debug_angles = analyzer.process_frame(
                model=model,
                label_mapping=label_mapping,
                landmarks=results.pose_landmarks,
                current_exercise=current_exercise
            )

        # --- UI Drawing ---
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Set status color
        if "Wrong" in form_status or "Still" in form_status:
            status_color = (0, 0, 200) # Red
        elif "Good" in form_status:
            status_color = (0, 180, 0) # Green
        else:
            status_color = (200, 100, 0) # Blue

        # Draw UI Elements
        
        # Top Bar
        cv2.rectangle(image_rgb, (0, 0), (1280, 80), (20, 20, 20), -1)
        
        # FPS Counter
        cv2.putText(image_rgb, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Status Box
        cv2.putText(image_rgb, 'STATUS', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_rgb, stable_prediction, (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_COLOR, 2, cv2.LINE_AA)
        
        # Rep Counter Box (with dynamic centering)
        cv2.rectangle(image_rgb, (1080, 0), (1280, 80), (40, 40, 40), -1)
        cv2.putText(image_rgb, 'REPS', (1150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rep_text = str(rep_counter)
        (w, h), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(image_rgb, rep_text, (1180 - w//2, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Feedback Bar
        cv2.rectangle(image_rgb, (0, 660), (1280, 720), status_color, -1)
        cv2.putText(image_rgb, form_status, (15, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Debug View (Optional)
        # y_pos = 110
        # for name, angle in debug_angles.items():
        #     cv2.putText(image_rgb, f"{name}: {int(angle)}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        #     y_pos += 25
        
        # Convert back to BGR for OpenCV display
        cv2.imshow('AI Pose Corrector', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

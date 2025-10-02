import cv2
import mediapipe as mp
import numpy as np
import json
import tensorflow as tf
from collections import deque
import time
import os

# --- CONFIGURATION ---
# ROBUST FILE PATHS: Build paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(SCRIPT_DIR, 'training', 'exercise_classifier_lstm.h5')
LABEL_MAPPING_FILENAME = os.path.join(SCRIPT_DIR, 'training', 'label_mapping.json')

SEQUENCE_LENGTH = 90
CONF_THRESHOLD = 0.80 
STABILITY_FRAMES = 10
UI_COLOR = (0, 150, 255)
# PERFORMANCE ENHANCEMENT: Predict every N frames
PREDICTION_INTERVAL = 3

# --- FINAL, STANDARDIZED HELPER FUNCTIONS ---

def calculate_angle(a, b, c):
    """Calculates the angle between three 3D landmark points."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_angle_features_for_model(landmarks):
    """Extracts the 8 key angles using the full 3D landmark data, matching the training script."""
    lm = landmarks
    mp_pose = mp.solutions.pose.PoseLandmark

    return np.array([
        calculate_angle(lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_ELBOW], lm[mp_pose.LEFT_WRIST]),
        calculate_angle(lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_ELBOW], lm[mp_pose.RIGHT_WRIST]),
        calculate_angle(lm[mp_pose.LEFT_ELBOW], lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_HIP]),
        calculate_angle(lm[mp_pose.RIGHT_ELBOW], lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_HIP]),
        calculate_angle(lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_HIP], lm[mp_pose.LEFT_KNEE]),
        calculate_angle(lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_HIP], lm[mp_pose.RIGHT_KNEE]),
        calculate_angle(lm[mp_pose.LEFT_HIP], lm[mp_pose.LEFT_KNEE], lm[mp_pose.LEFT_ANKLE]),
        calculate_angle(lm[mp_pose.RIGHT_HIP], lm[mp_pose.RIGHT_KNEE], lm[mp_pose.RIGHT_ANKLE])
    ])

def calculate_angle_2d(a, b, c):
    """This 2D version is kept for the form checker, which uses 2D coordinate arrays."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- FINAL CALIBRATED Exercise Analysis Class with Full Form Correction ---
class ExerciseAnalyzer:
    def __init__(self, reset_timeout=5.0):
        self.rep_counter = 0
        self.stage = None
        self.form_status = "START EXERCISE"
        self.status_color = (0, 255, 0)
        self.previous_exercise = "neutral"
        self.last_rep_time = time.time()
        self.RESET_TIMEOUT = reset_timeout
        self.debug_angles = {} # For the debug view

    def analyze_frame(self, exercise_name, landmarks):
        if landmarks is None:
            if exercise_name == "neutral":
                self.previous_exercise = "neutral"
                self.stage = None
            self.debug_angles.clear()
            return

        if exercise_name != self.previous_exercise:
            self.rep_counter = 0
            self.stage = None
            self.previous_exercise = exercise_name
            self.last_rep_time = time.time()

        self.form_status = "CORRECT FORM"
        self.status_color = (0, 255, 0)

        if time.time() - self.last_rep_time > self.RESET_TIMEOUT and self.stage is not None:
            self.stage = None
            self.form_status = "INACTIVE - RESET"
        
        mp_lm = mp.solutions.pose.PoseLandmark
        
        # ENHANCEMENT: Wrap all exercise logic in a try-except for robustness
        try:
            if 'bicepCurl' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]
                
                elbow_angle, shoulder_angle = 0, 0
                if left_wrist_lm.visibility > right_wrist_lm.visibility:
                    elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
                    shoulder_angle = calculate_angle_2d([left_elbow_lm.x, left_elbow_lm.y], [left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y])
                else:
                    elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])
                    shoulder_angle = calculate_angle_2d([right_elbow_lm.x, right_elbow_lm.y], [right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y])
                self.debug_angles = {'Elbow': elbow_angle, 'Shoulder': shoulder_angle}

                if elbow_angle > 150: self.stage = "down"
                if elbow_angle < 45 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if shoulder_angle > 45:
                    self.form_status = "ERROR: KEEP ELBOWS PINNED"; self.status_color = (0, 0, 255)

            elif 'shoulderPress' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]

                left_elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
                right_elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])
                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
                self.debug_angles = {'Avg Elbow': avg_elbow_angle}

                if avg_elbow_angle < 95: self.stage = "down"
                if avg_elbow_angle > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if self.stage == 'down':
                    left_shoulder_angle = calculate_angle_2d([left_elbow_lm.x, left_elbow_lm.y], [left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y])
                    right_shoulder_angle = calculate_angle_2d([right_elbow_lm.x, right_elbow_lm.y], [right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y])
                    avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
                    self.debug_angles['Avg Shoulder'] = avg_shoulder_angle
                    if avg_shoulder_angle < 30 or avg_shoulder_angle > 60:
                        self.form_status = "WARNING: TUCK ELBOWS AT 45 DEG"; self.status_color = (0, 165, 255)

            elif 'lateralRaise' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]; left_knee_lm = landmarks[mp_lm.LEFT_KNEE.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]; right_knee_lm = landmarks[mp_lm.RIGHT_KNEE.value]

                shoulder_angle = 0
                if left_elbow_lm.visibility > right_elbow_lm.visibility:
                    shoulder_angle = calculate_angle_2d([left_elbow_lm.x, left_elbow_lm.y], [left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y])
                else:
                    shoulder_angle = calculate_angle_2d([right_elbow_lm.x, right_elbow_lm.y], [right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y])
                
                if shoulder_angle < 30: self.stage = "down"
                if shoulder_angle > 90: self.form_status = "WARNING: DO NOT OVER-RAISE"; self.status_color = (0, 165, 255)
                if shoulder_angle > 75 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()

                torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
                self.debug_angles = {'Shoulder': shoulder_angle, 'Torso': torso_angle}
                if torso_angle < 160:
                    self.form_status = "ERROR: KEEP TORSO UPRIGHT"; self.status_color = (0, 0, 255)

            elif 'tricepKickback' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]; left_knee_lm = landmarks[mp_lm.LEFT_KNEE.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]; right_knee_lm = landmarks[mp_lm.RIGHT_KNEE.value]
                
                elbow_angle = 0
                if left_wrist_lm.visibility > right_wrist_lm.visibility:
                    elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
                else:
                    elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])
                
                if elbow_angle < 100: self.stage = "in"
                if elbow_angle > 160 and self.stage == 'in':
                    self.stage = "out"; self.rep_counter += 1; self.last_rep_time = time.time()

                if self.stage is not None:
                    # --- BUG FIX START ---
                    left_torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
                    right_torso_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y], [right_knee_lm.x, right_knee_lm.y])
                    torso_angle = left_torso_angle if left_hip_lm.visibility > right_hip_lm.visibility else right_torso_angle
                    self.debug_angles = {'Elbow': elbow_angle, 'Torso': torso_angle}
                    # --- BUG FIX END ---
                    
                    if torso_angle > 135:
                        self.form_status = "ERROR: BEND OVER MORE (45 DEG)"; self.status_color = (0, 0, 255)
                    elif self.stage == 'out' and elbow_angle < 160:
                        self.form_status = "EXTEND ARM FULLY"; self.status_color = (0, 165, 255)

            elif 'bentOverRow' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]; left_knee_lm = landmarks[mp_lm.LEFT_KNEE.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]; right_knee_lm = landmarks[mp_lm.RIGHT_KNEE.value]
                
                elbow_angle = 0
                if left_wrist_lm.visibility > right_wrist_lm.visibility:
                    elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
                else:
                    elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])

                if elbow_angle > 140: self.stage = "down"
                if elbow_angle < 70 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if self.stage is not None:
                    # --- BUG FIX START ---
                    left_torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
                    right_torso_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y], [right_knee_lm.x, right_knee_lm.y])
                    torso_angle = left_torso_angle if left_hip_lm.visibility > right_hip_lm.visibility else right_torso_angle
                    self.debug_angles = {'Elbow': elbow_angle, 'Torso': torso_angle}
                    # --- BUG FIX END ---
                    
                    if torso_angle > 135:
                        self.form_status = "ERROR: BEND OVER MORE (45 DEG)"; self.status_color = (0, 0, 255)
        except Exception as e:
            print(f"Error during form analysis for {exercise_name}: {e}")
            self.form_status = "ERROR: TRACKING LOST"
            self.status_color = (0,0,255)
            self.debug_angles.clear()

    def get_status(self):
        return self.rep_counter, self.form_status, self.status_color, self.debug_angles
    
# --- MAIN LOGIC ---
try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
    with open(LABEL_MAPPING_FILENAME, 'r') as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}
    print("✅ LSTM model and label mapping loaded.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

window_name = 'AI Fitness Trainer'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=STABILITY_FRAMES)
stable_exercise = "neutral"
analyzer = ExerciseAnalyzer()
frame_count = 0

print("--- Live feed started. Press 'q' to quit. ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        angle_features = extract_angle_features_for_model(results.pose_landmarks.landmark)
        
        if not np.any(np.isnan(angle_features)):
            sequence_buffer.append(angle_features)

            if len(sequence_buffer) == SEQUENCE_LENGTH and frame_count % PREDICTION_INTERVAL == 0:
                input_data = np.expand_dims(np.array(sequence_buffer), axis=0)
                prediction_probs = model.predict(input_data, verbose=0)[0]
                predicted_index = np.argmax(prediction_probs)
                predicted_class = label_mapping.get(predicted_index, "unknown")
                current_confidence = prediction_probs[predicted_index]

                if current_confidence >= CONF_THRESHOLD:
                    prediction_buffer.append(predicted_class)
                    if len(set(prediction_buffer)) == 1 and len(prediction_buffer) == STABILITY_FRAMES:
                        stable_exercise = prediction_buffer[0]
                else:
                    prediction_buffer.clear()
                    stable_exercise = "neutral"

        analyzer.analyze_frame(stable_exercise, results.pose_landmarks.landmark)
    else:
        sequence_buffer.clear()
        prediction_buffer.clear()
        stable_exercise = "neutral"
        analyzer.analyze_frame("neutral", None)

    rep_counter, form_status, status_color, debug_angles = analyzer.get_status()
    
    # --- UI DRAWING ---
    # Draw UI elements first
    cv2.rectangle(image_rgb, (0, 0), (1280, 70), UI_COLOR, -1)
    cv2.putText(image_rgb, 'EXERCISE', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image_rgb, stable_exercise.replace('_', ' ').title(), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(image_rgb, 'REPS', (1150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image_rgb, str(rep_counter), (1140, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(image_rgb, (0, 660), (1280, 720), status_color, -1)
    cv2.putText(image_rgb, form_status, (15, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Improved Debug View
    y_pos = 110
    for name, angle in debug_angles.items():
        cv2.putText(image_rgb, f"{name}: {int(angle)}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos += 30

    # Draw landmarks on top of the UI
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, image_bgr)
    
    frame_count += 1
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

import cv2
import mediapipe as mp
import numpy as np
import json
import tensorflow as tf
from collections import deque
import time

# --- CONFIGURATION ---
MODEL_FILENAME = 'exercise_classifier_lstm.h5'
LABEL_MAPPING_FILENAME = 'label_mapping.json'
SEQUENCE_LENGTH = 90
CONF_THRESHOLD = 0.80 
STABILITY_FRAMES = 10
UI_COLOR = (0, 150, 255)

# --- HELPER FUNCTIONS ---
def calculate_angle_2d(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_angle_features_for_model(landmarks):
    lm = landmarks
    mp_pose = mp.solutions.pose.PoseLandmark

    left_elbow = calculate_angle_2d([lm[mp_pose.LEFT_SHOULDER].x, lm[mp_pose.LEFT_SHOULDER].y], [lm[mp_pose.LEFT_ELBOW].x, lm[mp_pose.LEFT_ELBOW].y], [lm[mp_pose.LEFT_WRIST].x, lm[mp_pose.LEFT_WRIST].y])
    right_elbow = calculate_angle_2d([lm[mp_pose.RIGHT_SHOULDER].x, lm[mp_pose.RIGHT_SHOULDER].y], [lm[mp_pose.RIGHT_ELBOW].x, lm[mp_pose.RIGHT_ELBOW].y], [lm[mp_pose.RIGHT_WRIST].x, lm[mp_pose.RIGHT_WRIST].y])
    left_shoulder = calculate_angle_2d([lm[mp_pose.LEFT_ELBOW].x, lm[mp_pose.LEFT_ELBOW].y], [lm[mp_pose.LEFT_SHOULDER].x, lm[mp_pose.LEFT_SHOULDER].y], [lm[mp_pose.LEFT_HIP].x, lm[mp_pose.LEFT_HIP].y])
    right_shoulder = calculate_angle_2d([lm[mp_pose.RIGHT_ELBOW].x, lm[mp_pose.RIGHT_ELBOW].y], [lm[mp_pose.RIGHT_SHOULDER].x, lm[mp_pose.RIGHT_SHOULDER].y], [lm[mp_pose.RIGHT_HIP].x, lm[mp_pose.RIGHT_HIP].y])
    left_hip = calculate_angle_2d([lm[mp_pose.LEFT_SHOULDER].x, lm[mp_pose.LEFT_SHOULDER].y], [lm[mp_pose.LEFT_HIP].x, lm[mp_pose.LEFT_HIP].y], [lm[mp_pose.LEFT_KNEE].x, lm[mp_pose.LEFT_KNEE].y])
    right_hip = calculate_angle_2d([lm[mp_pose.RIGHT_SHOULDER].x, lm[mp_pose.RIGHT_SHOULDER].y], [lm[mp_pose.RIGHT_HIP].x, lm[mp_pose.RIGHT_HIP].y], [lm[mp_pose.RIGHT_KNEE].x, lm[mp_pose.RIGHT_KNEE].y])
    left_knee = calculate_angle_2d([lm[mp_pose.LEFT_HIP].x, lm[mp_pose.LEFT_HIP].y], [lm[mp_pose.LEFT_KNEE].x, lm[mp_pose.LEFT_KNEE].y], [lm[mp_pose.LEFT_ANKLE].x, lm[mp_pose.LEFT_ANKLE].y])
    right_knee = calculate_angle_2d([lm[mp_pose.RIGHT_HIP].x, lm[mp_pose.RIGHT_HIP].y], [lm[mp_pose.RIGHT_KNEE].x, lm[mp_pose.RIGHT_KNEE].y], [lm[mp_pose.RIGHT_ANKLE].x, lm[mp_pose.RIGHT_ANKLE].y])
    
    return np.array([left_elbow, right_elbow, left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee])

# --- FINAL UPGRADED Exercise Analysis Class with Full Form Correction ---
class ExerciseAnalyzer:
    def __init__(self, reset_timeout=5.0):
        self.rep_counter = 0
        self.stage = None
        self.form_status = "START EXERCISE"
        self.status_color = (0, 255, 0)
        self.previous_exercise = "neutral"
        self.last_rep_time = time.time()
        self.RESET_TIMEOUT = reset_timeout

    def analyze_frame(self, exercise_name, landmarks):
        if landmarks is None:
            if exercise_name == "neutral":
                self.previous_exercise = "neutral"
                self.stage = None
            return

        if exercise_name != self.previous_exercise:
            self.rep_counter = 0
            self.stage = None
            self.previous_exercise = exercise_name
            self.last_rep_time = time.time()

        if time.time() - self.last_rep_time > self.RESET_TIMEOUT and self.stage is not None:
            self.stage = None
            self.form_status = "INACTIVE - RESET"
        
        if self.form_status != "INACTIVE - RESET":
            self.form_status = "CORRECT FORM"
        self.status_color = (0, 255, 0)
        
        mp_lm = mp.solutions.pose.PoseLandmark

        if 'bicepCurl' in exercise_name:
            left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]
            left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]
            left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]
            left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]
            right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]
            right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]
            right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]
            right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]
            
            left_wrist_visibility = left_wrist_lm.visibility
            right_wrist_visibility = right_wrist_lm.visibility
            
            elbow_angle = 0
            if left_wrist_visibility > right_wrist_visibility:
                elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
            else:
                elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])

            if elbow_angle > 150: self.stage = "down"
            if elbow_angle < 45 and self.stage == 'down':
                self.stage = "up"
                self.rep_counter += 1
                self.last_rep_time = time.time()
            
            shoulder_angle = 0
            if left_wrist_visibility > right_wrist_visibility:
                shoulder_angle = calculate_angle_2d([left_elbow_lm.x, left_elbow_lm.y], [left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y])
            else:
                shoulder_angle = calculate_angle_2d([right_elbow_lm.x, right_elbow_lm.y], [right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y])

            if shoulder_angle > 45:
                self.form_status = "ERROR: KEEP ELBOWS PINNED"
                self.status_color = (0, 0, 255)

        elif 'shoulderPress' in exercise_name:
            left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]
            left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]
            left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]
            right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]
            right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]
            right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]

            left_elbow_visibility = left_elbow_lm.visibility
            right_elbow_visibility = right_elbow_lm.visibility
            
            elbow_angle = 0
            if left_elbow_visibility > right_elbow_visibility:
                elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
            else:
                elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])

            if elbow_angle < 95: self.stage = "down"
            if elbow_angle > 160 and self.stage == 'down':
                self.stage = "up"
                self.rep_counter += 1
                self.last_rep_time = time.time()
            
            if self.stage == 'up' and elbow_angle > 95:
                self.form_status = "LOWER THE WEIGHT MORE"
                self.status_color = (0, 165, 255)

        elif 'lateralRaise' in exercise_name:
            left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]
            left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]
            left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]
            left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]
            right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]
            right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]
            right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]
            right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]

            left_elbow_visibility = left_elbow_lm.visibility
            right_elbow_visibility = right_elbow_lm.visibility

            shoulder_angle = 0
            if left_elbow_visibility > right_elbow_visibility:
                shoulder_angle = calculate_angle_2d([left_elbow_lm.x, left_elbow_lm.y], [left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y])
            else:
                shoulder_angle = calculate_angle_2d([right_elbow_lm.x, right_elbow_lm.y], [right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y])
            
            if shoulder_angle < 30: self.stage = "down"
            if shoulder_angle > 75 and self.stage == 'down':
                self.stage = "up"
                self.rep_counter += 1
                self.last_rep_time = time.time()

            elbow_angle = 0
            if left_elbow_visibility > right_elbow_visibility:
                elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
            else:
                elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])

            if elbow_angle < 140:
                self.form_status = "ERROR: KEEP ARMS STRAIGHTER"
                self.status_color = (0, 0, 255)
        
        elif 'tricepKickback' in exercise_name:
            left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]
            left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]
            left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]
            left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]
            left_knee_lm = landmarks[mp_lm.LEFT_KNEE.value]
            right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]
            right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]
            right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]
            right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]
            right_knee_lm = landmarks[mp_lm.RIGHT_KNEE.value]

            left_wrist_visibility = left_wrist_lm.visibility
            right_wrist_visibility = right_wrist_lm.visibility
            
            elbow_angle = 0
            if left_wrist_visibility > right_wrist_visibility:
                elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
            else:
                elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])
            
            if elbow_angle < 100: self.stage = "in"
            if elbow_angle > 160 and self.stage == 'in':
                self.stage = "out"
                self.rep_counter += 1
                self.last_rep_time = time.time()

            torso_angle = 0
            if landmarks[mp_lm.LEFT_HIP.value].visibility > landmarks[mp_lm.RIGHT_HIP.value].visibility:
                 torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
            else: # Use right side landmarks if they are more visible
                 torso_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y], [right_knee_lm.x, right_knee_lm.y])

            if torso_angle > 140:
                self.form_status = "ERROR: BEND YOUR TORSO MORE"
                self.status_color = (0, 0, 255)
            elif self.stage == 'out' and elbow_angle < 160:
                self.form_status = "EXTEND ARM FULLY"
                self.status_color = (0, 165, 255)

        elif 'bentOverRow' in exercise_name:
            left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]
            left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]
            left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]
            left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]
            left_knee_lm = landmarks[mp_lm.LEFT_KNEE.value]
            right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]
            right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]
            right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]
            right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]
            right_knee_lm = landmarks[mp_lm.RIGHT_KNEE.value]
            
            left_wrist_visibility = left_wrist_lm.visibility
            right_wrist_visibility = right_wrist_lm.visibility
            
            elbow_angle = 0
            if left_wrist_visibility > right_wrist_visibility:
                elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
            else:
                elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])

            if elbow_angle > 140: self.stage = "down"
            if elbow_angle < 70 and self.stage == 'down':
                self.stage = "up"
                self.rep_counter += 1
                self.last_rep_time = time.time()
            
            torso_angle = 0
            if landmarks[mp_lm.LEFT_HIP.value].visibility > landmarks[mp_lm.RIGHT_HIP.value].visibility:
                torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
            else:
                torso_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y], [right_knee_lm.x, right_knee_lm.y])
            
            if torso_angle > 100:
                self.form_status = "ERROR: STAY BENT OVER"
                self.status_color = (0, 0, 255)

    def get_status(self):
        return self.rep_counter, self.form_status, self.status_color

# --- MAIN LOGIC ---
try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
    with open(LABEL_MAPPING_FILENAME, 'r') as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}
    print("✅ LSTM model and label mapping loaded.")
except Exception as e:
    print(f"Error loading assets: {e}")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

window_name = 'AI Fitness Trainer'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=STABILITY_FRAMES)
stable_exercise = "neutral"
analyzer = ExerciseAnalyzer()

print("--- Live feed started. Press 'q' to quit. ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        angle_features = extract_angle_features_for_model(results.pose_landmarks.landmark)
        
        if not np.any(np.isnan(angle_features)):
            sequence_buffer.append(angle_features)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence_buffer), axis=0)
                
                prediction_probs = model.predict(input_data, verbose=0)[0]
                predicted_index = np.argmax(prediction_probs)
                
                # --- CORRECTED: Simplified and safer label lookup ---
                predicted_class = label_mapping.get(predicted_index, "unknown")
                current_confidence = prediction_probs[predicted_index]

                if current_confidence >= CONF_THRESHOLD:
                    prediction_buffer.append(predicted_class)
                    if len(prediction_buffer) == STABILITY_FRAMES and len(set(prediction_buffer)) == 1:
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

    rep_counter, form_status, status_color = analyzer.get_status()
    
    cv2.rectangle(image, (0, 0), (450, 70), UI_COLOR, -1)
    cv2.putText(image, 'EXERCISE', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, stable_exercise.replace('_', ' ').title(), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.rectangle(image, (image.shape[1] - 200, 0), (image.shape[1], 70), UI_COLOR, -1)
    cv2.putText(image, 'REPS', (image.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, str(rep_counter), (image.shape[1] - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(image, (0, image.shape[0] - 60), (image.shape[1], image.shape[0]), status_color, -1)
    cv2.putText(image, form_status, (15, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
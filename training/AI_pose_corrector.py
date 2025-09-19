# --- IMPORTS ---
import cv2
import mediapipe as mp
import numpy as np
import json
import tensorflow as tf
from collections import deque

# --- CONFIGURATION ---
MODEL_FILENAME = 'exercise_classifier_lstm.h5'
LABEL_MAPPING_FILENAME = 'label_mapping.json'
SEQUENCE_LENGTH = 90  # Number of frames for one sequence (must match training)
CONF_THRESHOLD = 0.80 # Confidence threshold for displaying a prediction
STABILITY_FRAMES = 10   # Number of consistent frames to consider a prediction stable
UI_COLOR = (0, 150, 255) # A new color for the UI

# --- HELPER FUNCTIONS (Kept for ExerciseAnalyzer) ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def normalize_pose_robust(landmarks):
    """Robust pose normalization using hip-to-shoulder torso length."""
    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    if np.isnan(landmarks_np).any():
        return np.full((33, 3), np.nan)

    left_hip = landmarks_np[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks_np[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    hip_center = (left_hip + right_hip) / 2.0
    
    left_shoulder = landmarks_np[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks_np[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    torso_length = np.linalg.norm(hip_center - shoulder_center)
    if torso_length < 1e-6:
        return np.full((33, 3), np.nan)

    normalized_landmarks = (landmarks_np - hip_center) / torso_length
    return normalized_landmarks

# --- Exercise Analysis Class (Kept from original for rep counting/form) ---
class ExerciseAnalyzer:
    def __init__(self):
        self.rep_counter = 0
        self.stage = None
        self.form_status = "START EXERCISE"
        self.status_color = (0, 255, 0)
        self.previous_exercise = "neutral"

    def analyze_frame(self, exercise_name, landmarks):
        if exercise_name != self.previous_exercise:
            self.rep_counter = 0
            self.stage = None
            self.previous_exercise = exercise_name
        
        # Get coordinates for angle calculations
        shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Default form status
        self.form_status = "CORRECT FORM"
        self.status_color = (0, 255, 0)

        if 'bicepCurl' in exercise_name:
            angle = calculate_angle(shoulder, elbow, wrist)
            if angle > 160:
                self.stage = "down"
            if angle < 30 and self.stage == 'down':
                self.stage = "up"
                self.rep_counter += 1
        
        elif 'squat' in exercise_name:
            angle = calculate_angle(hip, knee, ankle)
            if angle > 160:
                self.stage = "up"
            if angle < 90 and self.stage == 'up':
                self.stage = "down"
                self.rep_counter += 1

    def get_status(self):
        return self.rep_counter, self.form_status, self.status_color

# --- MAIN LOGIC ---
try:
    # Load the trained LSTM model and label mapping
    model = tf.keras.models.load_model(MODEL_FILENAME)
    with open(LABEL_MAPPING_FILENAME, 'r') as f:
        label_mapping = json.load(f)
    print("✅ LSTM model and label mapping loaded.")
except Exception as e:
    print(f"Error loading assets: {e}")
    exit()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- NEW CODE TO PRESERVE ASPECT RATIO ---
# Get the native resolution from the camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the desired window width, and calculate the height to maintain aspect ratio
window_width = 1600
aspect_ratio = frame_height / frame_width
window_height = int(window_width * aspect_ratio)

# Create and resize the window
window_name = 'AI Fitness Trainer'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, window_width, window_height)

# --- Initialize variables for sequence prediction ---
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=STABILITY_FRAMES)
stable_exercise = "neutral"
current_confidence = 0.0
analyzer = ExerciseAnalyzer()

print("--- Live feed started. Press 'q' to quit. ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Process frame with MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # --- PREDICTION LOGIC ---
        # Normalize landmarks for model input
        normalized_landmarks = normalize_pose_robust(results.pose_landmarks.landmark)
        
        if not np.isnan(normalized_landmarks).any():
            # Flatten and add to sequence buffer
            sequence_buffer.append(normalized_landmarks.flatten())

            # Predict if the buffer is full
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence_buffer), axis=0)
                
                # Get model prediction
                prediction_probs = model.predict(input_data)[0]
                predicted_index = np.argmax(prediction_probs)
                current_confidence = prediction_probs[predicted_index]
                
                # The keys in JSON are strings, so convert index to string
                predicted_class = label_mapping[str(predicted_index)]

                # Stability logic
                if current_confidence >= CONF_THRESHOLD:
                    prediction_buffer.append(predicted_class)
                    if len(prediction_buffer) == STABILITY_FRAMES and len(set(prediction_buffer)) == 1:
                        stable_exercise = prediction_buffer[0]
                else:
                    prediction_buffer.clear()

        # --- REP COUNTING & FORM ANALYSIS ---
        analyzer.analyze_frame(stable_exercise, results.pose_landmarks.landmark)

    else:
        # If no landmarks, clear buffers
        sequence_buffer.clear()
        prediction_buffer.clear()
        stable_exercise = "neutral"

    # --- DISPLAY UI ---
    rep_counter, form_status, status_color = analyzer.get_status()
    
    # Status bar for exercise
    cv2.rectangle(image, (0, 0), (450, 70), UI_COLOR, -1)
    cv2.putText(image, 'EXERCISE', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, stable_exercise.replace('_', ' ').title(), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Status bar for reps
    cv2.rectangle(image, (image.shape[1] - 200, 0), (image.shape[1], 70), UI_COLOR, -1)
    cv2.putText(image, 'REPS', (image.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, str(rep_counter), (image.shape[1] - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Status bar for form feedback
    cv2.rectangle(image, (0, image.shape[0] - 60), (image.shape[1], image.shape[0]), status_color, -1)
    cv2.putText(image, form_status, (15, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('AI Fitness Trainer', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
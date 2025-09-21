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
SEQUENCE_LENGTH = 90      # Number of frames for one sequence (must match training)
CONF_THRESHOLD = 0.80     # Confidence threshold for displaying a prediction
STABILITY_FRAMES = 10     # Number of consistent frames to consider a prediction stable
UI_COLOR = (0, 150, 255)  # A new color for the UI

# --- NEW HELPER FUNCTIONS ---
def normalize_landmarks(landmarks):
    """
    Normalizes pose landmarks based on the bounding box of the pose,
    making it independent of camera distance and body orientation.
    Returns a (33, 2) numpy array with normalized x and y coordinates.
    """
    landmarks_np = np.array([[lm.x, lm.y] for lm in landmarks])

    min_coords = np.min(landmarks_np, axis=0)
    max_coords = np.max(landmarks_np, axis=0)
    
    center = (min_coords + max_coords) / 2.0
    scale = np.max(max_coords - min_coords)
    
    if scale < 1e-6:
        return np.full((33, 2), np.nan) # Avoid division by zero
        
    normalized_landmarks = (landmarks_np - center) / scale
    return normalized_landmarks

def calculate_angle_2d(a, b, c):
    """Calculates the angle between three 2D points (as numpy arrays)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- UPDATED Exercise Analysis Class ---
class ExerciseAnalyzer:
    def __init__(self):
        self.rep_counter = 0
        self.stage = None
        self.form_status = "START EXERCISE"
        self.status_color = (0, 255, 0)
        self.previous_exercise = "neutral"

    def analyze_frame(self, exercise_name, normalized_landmarks):
        # Reset counter if exercise changes
        if exercise_name != self.previous_exercise:
            self.rep_counter = 0
            self.stage = None
            self.previous_exercise = exercise_name
        
        # Default form status
        self.form_status = "CORRECT FORM"
        self.status_color = (0, 255, 0)

        # Landmarks mapping
        mp_lm = mp.solutions.pose.PoseLandmark

        if 'bicepCurl' in exercise_name:
            shoulder = normalized_landmarks[mp_lm.LEFT_SHOULDER.value]
            elbow = normalized_landmarks[mp_lm.LEFT_ELBOW.value]
            wrist = normalized_landmarks[mp_lm.LEFT_WRIST.value]
            
            angle = calculate_angle_2d(shoulder, elbow, wrist)
            
            if angle > 160:
                self.stage = "down"
            if angle < 30 and self.stage == 'down':
                self.stage = "up"
                self.rep_counter += 1
        
        elif 'squat' in exercise_name:
            hip = normalized_landmarks[mp_lm.LEFT_HIP.value]
            knee = normalized_landmarks[mp_lm.LEFT_KNEE.value]
            ankle = normalized_landmarks[mp_lm.LEFT_ANKLE.value]

            angle = calculate_angle_2d(hip, knee, ankle)
            
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

# Set HD resolution and create a resizable window
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Attempted 1920x1080, camera provided: {actual_width}x{actual_height}")

window_name = 'AI Fitness Trainer'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1600, 900)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables for sequence prediction
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
        
        # --- NORMALIZATION AND PREDICTION LOGIC ---
        normalized_landmarks = normalize_landmarks(results.pose_landmarks.landmark)
        
        if not np.isnan(normalized_landmarks).any():
            # Flatten for model input and add to sequence buffer
            sequence_buffer.append(normalized_landmarks.flatten())

            # Predict if the buffer is full
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence_buffer), axis=0)
                
                prediction_probs = model.predict(input_data, verbose=0)[0]
                predicted_index = np.argmax(prediction_probs)
                current_confidence = prediction_probs[predicted_index]
                
                predicted_class = label_mapping[str(predicted_index)]

                # Stability logic
                if current_confidence >= CONF_THRESHOLD:
                    prediction_buffer.append(predicted_class)
                    if len(prediction_buffer) == STABILITY_FRAMES and len(set(prediction_buffer)) == 1:
                        stable_exercise = prediction_buffer[0]
                else:
                    prediction_buffer.clear()

        # --- REP COUNTING & FORM ANALYSIS ---
        # Pass the normalized landmarks to the analyzer
        analyzer.analyze_frame(stable_exercise, normalized_landmarks)

    else:
        # If no landmarks, clear buffers and reset state
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

    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
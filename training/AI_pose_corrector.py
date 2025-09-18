# --- IMPORTS (same as original) ---
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION (same as original) ---
MODEL_FILENAME = 'exercise_model_mlp.pkl'
SCALER_FILENAME = 'scaler.pkl'
FEATURE_NAMES_FILENAME = 'feature_names.pkl'
CONF_THRESHOLD = 0.60 
STABILITY_FRAMES = 10 
UI_COLOR = (245, 117, 16)

# --- HELPER FUNCTIONS (Corrected) ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    dot_product = np.dot(ba, bc)

    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def normalize_pose_robust(landmarks, mp_pose_module):
    """Robust pose normalization using hip-to-shoulder torso length."""
    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # Handle cases where input landmarks might contain NaNs
    if np.isnan(landmarks_np).any():
        return np.full((33, 3), np.nan), np.nan

    left_hip = landmarks_np[mp_pose_module.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks_np[mp_pose_module.PoseLandmark.RIGHT_HIP.value]
    hip_center = (left_hip + right_hip) / 2.0
    
    left_shoulder = landmarks_np[mp_pose_module.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks_np[mp_pose_module.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    torso_length = np.linalg.norm(hip_center - shoulder_center) + 1e-6

    if torso_length < 1e-5:
        return np.full((33, 3), np.nan), np.nan

    normalized_landmarks = (landmarks_np - hip_center) / torso_length
    
    return normalized_landmarks, hip_center

# --- Exercise Analysis Class (same as original) ---
class ExerciseAnalyzer:
    def __init__(self):
        self.rep_counter = 0
        self.stage = None
        self.form_status = "CORRECT FORM"
        self.status_color = (0, 255, 0)
        self.previous_exercise = "UNKNOWN"

    def analyze_frame(self, exercise_name, angles, distances):
        # Reset counters if the exercise changes
        if exercise_name != self.previous_exercise:
            self.rep_counter = 0
            self.stage = None
            self.form_status = "CORRECT FORM"
            self.status_color = (0, 255, 0)
            self.previous_exercise = exercise_name
            return

        self.form_status = "CORRECT FORM"
        self.status_color = (0, 255, 0)

        angle_left_elbow = angles.get('angle_left_elbow')
        angle_left_knee = angles.get('angle_left_knee')
        angle_left_hip = angles.get('angle_left_hip')
        dist_y_l_wrist_shoulder = distances.get('dist_y_l_wrist_shoulder')

        if 'squat' in exercise_name:
            if angle_left_knee < 100:
                self.stage = "down"
            if angle_left_knee > 160 and self.stage == "down":
                self.stage = "up"
                self.rep_counter += 1
            if angle_left_hip < 90:
                self.form_status = "GO DEEPER"; self.status_color = (0, 165, 255)

        elif 'bicep_curl' in exercise_name:
            if angle_left_elbow < 40:
                self.stage = "up"
            if angle_left_elbow > 160 and self.stage == "up":
                self.stage = "down"
                self.rep_counter += 1
            if dist_y_l_wrist_shoulder > 0.1:
                self.form_status = "KEEP ELBOWS IN"; self.status_color = (0, 0, 255)

        # Add logic for other exercises here following the same pattern
        # e.g., for bench press, use elbow angles
        elif 'bench_press' in exercise_name:
            if angle_left_elbow > 160: self.stage = "up"
            if angle_left_elbow < 90 and self.stage == 'up':
                self.stage = "down"; self.rep_counter += 1

        # etc.
    
    def get_status(self):
        return self.rep_counter, self.form_status, self.status_color

# --- MAIN LOGIC ---
try:
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILENAME, 'rb') as f:
        scaler = pickle.load(f)
    with open(FEATURE_NAMES_FILENAME, 'rb') as f:
        feature_names = pickle.load(f)
    print("✅ All assets loaded.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure all files exist.")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

window_name = 'AI Fitness Trainer'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1600, 900)

prediction_buffer = []
stable_exercise = "UNKNOWN"
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
        try:
            landmarks_original = results.pose_landmarks.landmark
            
            # Unpack the returned values correctly
            normalized_coords_np, hip_center = normalize_pose_robust(landmarks_original, mp_pose)
            
            # Skip if normalization failed (produced NaNs)
            if np.isnan(normalized_coords_np).any(): 
                continue

            # Helper to get normalized 3D coordinates for a specific landmark
            def get_norm_lm_coords(lm_index):
                return normalized_coords_np[lm_index]
            
            # Define Key Landmarks from the normalized coordinates
            nose = get_norm_lm_coords(mp_pose.PoseLandmark.NOSE.value)
            left_ear = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_EAR.value)
            right_ear = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_EAR.value)
            
            left_shoulder = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            right_shoulder = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            left_hip = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_HIP.value)
            right_hip = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_HIP.value)
            
            left_elbow = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_ELBOW.value)
            left_wrist = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_WRIST.value)
            left_index = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_INDEX.value)
            right_elbow = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            right_wrist = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            right_index = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_INDEX.value)

            left_knee = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_KNEE.value)
            left_ankle = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_ANKLE.value)
            left_heel = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_HEEL.value)
            right_knee = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value)
            right_ankle = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
            right_heel = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_HEEL.value)

            # --- Calculate all new angles ---
            angles = {
                'angle_torso_side_left': calculate_angle(left_shoulder, left_hip, left_knee),
                'angle_torso_side_right': calculate_angle(right_shoulder, right_hip, right_knee),
                'angle_torso_front': calculate_angle(left_shoulder, right_hip, right_shoulder),
                'angle_neck': calculate_angle(nose, left_ear, right_ear),
                'angle_spine_hip_shoulder_left': calculate_angle(left_hip, left_shoulder, right_shoulder),
                'angle_spine_hip_shoulder_right': calculate_angle(right_hip, right_shoulder, left_shoulder),
                'angle_left_elbow': calculate_angle(left_shoulder, left_elbow, left_wrist),
                'angle_right_elbow': calculate_angle(right_shoulder, right_elbow, right_wrist),
                'angle_left_shoulder_abduction': calculate_angle(left_hip, left_shoulder, left_elbow),
                'angle_right_shoulder_abduction': calculate_angle(right_hip, right_shoulder, right_elbow),
                'angle_left_wrist': calculate_angle(left_elbow, left_wrist, left_index),
                'angle_right_wrist': calculate_angle(right_elbow, right_wrist, right_index),
                'angle_left_hip': calculate_angle(left_shoulder, left_hip, left_knee),
                'angle_right_hip': calculate_angle(right_shoulder, right_hip, right_knee),
                'angle_left_knee': calculate_angle(left_hip, left_knee, left_ankle),
                'angle_right_knee': calculate_angle(right_hip, right_knee, right_ankle),
                'angle_left_ankle': calculate_angle(left_knee, left_ankle, left_heel),
                'angle_right_ankle': calculate_angle(right_knee, right_ankle, right_heel),
                'angle_shoulder_hip_twist_left': calculate_angle(right_shoulder, left_hip, right_hip),
                'angle_shoulder_hip_twist_right': calculate_angle(left_shoulder, right_hip, left_hip)
            }
            
            # --- Calculate all new distances ---
            distances = {
                'dist_shoulders': np.linalg.norm(left_shoulder - right_shoulder),
                'dist_hips': np.linalg.norm(left_hip - right_hip),
                'dist_left_wrist_knee': np.linalg.norm(left_wrist - left_knee),
                'dist_right_wrist_knee': np.linalg.norm(right_wrist - right_knee),
                'dist_left_elbow_hip': np.linalg.norm(left_elbow - left_hip),
                'dist_right_elbow_hip': np.linalg.norm(right_elbow - right_hip),
                'dist_left_ankle_wrist': np.linalg.norm(left_ankle - left_wrist),
                'dist_right_ankle_wrist': np.linalg.norm(right_ankle - right_wrist),
                'dist_nose_hip': np.linalg.norm(nose - hip_center),
                'dist_y_l_wrist_shoulder': abs(left_wrist[1] - left_shoulder[1]),
                'dist_y_r_wrist_shoulder': abs(right_wrist[1] - right_shoulder[1]),
                'dist_y_l_hip_knee': abs(left_hip[1] - left_knee[1]),
                'dist_y_r_hip_knee': abs(right_hip[1] - right_knee[1]),
                'dist_y_l_shoulder_hip': abs(left_shoulder[1] - left_hip[1]),
                'dist_y_r_shoulder_hip': abs(right_shoulder[1] - right_hip[1]),
                'dist_y_l_ankle_heel': abs(left_ankle[1] - left_heel[1]),
                'dist_y_r_ankle_heel': abs(right_ankle[1] - right_heel[1]),
                'dist_z_l_wrist_hip': abs(left_wrist[2] - left_hip[2]),
                'dist_z_r_wrist_hip': abs(right_wrist[2] - right_hip[2]),
                'dist_z_l_shoulder_hip': abs(left_shoulder[2] - left_hip[2]),
                'dist_z_r_shoulder_hip': abs(right_shoulder[2] - right_hip[2]),
                'dist_z_nose_hip': abs(nose[2] - hip_center[2])
            }

            # Create a dataframe for the model
            final_row_values = list(angles.values()) + list(distances.values())
            X_live = pd.DataFrame([final_row_values], columns=feature_names)
            X_scaled = scaler.transform(X_live)

            predicted_class = model.predict(X_scaled)[0]
            confidence = np.max(model.predict_proba(X_scaled))

            print(f"Predicted: {predicted_class:<30} | Confidence: {confidence:.2f}")

            if confidence >= CONF_THRESHOLD:
                prediction_buffer.append(predicted_class)
                if len(prediction_buffer) > STABILITY_FRAMES:
                    prediction_buffer.pop(0)

                if len(prediction_buffer) == STABILITY_FRAMES and len(set(prediction_buffer)) == 1:
                    stable_exercise = prediction_buffer[0]
            else:
                pass
            
            analyzer.analyze_frame(stable_exercise, angles, distances)
            rep_counter, form_status, status_color = analyzer.get_status()

        except (IndexError, TypeError) as e:
            pass
        except Exception as e:
            pass
            
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    cv2.rectangle(image, (0, 0), (450, 110), UI_COLOR, -1)
    cv2.putText(image, 'EXERCISE', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, stable_exercise.replace('_', ' ').title(), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'REPS', (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    rep_counter, form_status, status_color = analyzer.get_status()
    cv2.putText(image, str(rep_counter), (295, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, image.shape[0] - 60), (image.shape[1], image.shape[0]), status_color, -1)
    cv2.putText(image, form_status, (15, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd # NEW: Import pandas for creating DataFrame with feature names
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
MODEL_FILENAME = 'exercise_model_mlp.pkl'
SCALER_FILENAME = 'scaler.pkl'
FEATURE_NAMES_FILENAME = 'feature_names.pkl' # NEW: Filename for feature names
CONF_THRESHOLD = 0.30
STABILITY_FRAMES = 15
UI_COLOR = (245, 117, 16)

# --- Helper Function to Calculate Angles (Corrected 3D Version) ---
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

# --- Pose Normalization Function ---
def normalize_pose(landmarks):
    """Normalizes landmarks to be invariant to position and scale."""
    mp_pose = mp.solutions.pose

    # Ensure landmark objects are converted to numpy arrays for calculations
    left_hip_coords = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z])
    right_hip_coords = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])
    hip_center = (left_hip_coords + right_hip_coords) / 2.0

    left_shoulder_coords = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
    right_shoulder_coords = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z])
    torso_size = np.linalg.norm(left_shoulder_coords - right_shoulder_coords) + 1e-6

    normalized_landmarks = []
    for lm in landmarks:
        normalized_lm_coords = (np.array([lm.x, lm.y, lm.z]) - hip_center) / torso_size
        normalized_landmarks.append({
            'x': normalized_lm_coords[0],
            'y': normalized_lm_coords[1],
            'z': normalized_lm_coords[2]
        })

    return normalized_landmarks

# --- Load the Model, Scaler, and Feature Names ---
model = None
scaler = None
feature_names = None # NEW: Initialize feature_names

try:
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded successfully from '{MODEL_FILENAME}'")
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_FILENAME}'")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    with open(SCALER_FILENAME, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler loaded successfully from '{SCALER_FILENAME}'")
except FileNotFoundError:
    print(f"Error: Scaler file not found at '{SCALER_FILENAME}'. Make sure you run train_model.py.")
    exit()
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

try: # NEW: Load feature names
    with open(FEATURE_NAMES_FILENAME, 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✅ Feature names loaded successfully from '{FEATURE_NAMES_FILENAME}'")
except FileNotFoundError:
    print(f"Error: Feature names file not found at '{FEATURE_NAMES_FILENAME}'. Make sure you run train_model.py.")
    exit()
except Exception as e:
    print(f"Error loading feature names: {e}")
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

rep_counter = 0
exercise_stage = None
form_status = "CORRECT FORM"
status_color = (0, 255, 0)

prediction_buffer = []
stable_exercise = "UNKNOWN"

print("--- Live feed started. Press 'q' to quit. ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        try:
            landmarks_original = results.pose_landmarks.landmark
            landmarks_normalized = normalize_pose(landmarks_original)

            # Extract normalized coordinates for calculations
            def get_norm_lm_coords(lm_index):
                return [
                    landmarks_normalized[lm_index]['x'],
                    landmarks_normalized[lm_index]['y'],
                    landmarks_normalized[lm_index]['z']
                ]

            left_shoulder = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            left_elbow = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_ELBOW.value)
            left_wrist = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_WRIST.value)
            left_hip = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_HIP.value)
            left_knee = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_KNEE.value)
            left_ankle = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_ANKLE.value)

            right_shoulder = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            right_elbow = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            right_wrist = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            right_hip = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_HIP.value)
            right_knee = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value)
            right_ankle = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

            angle_left_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)
            angle_left_shoulder = calculate_angle(left_hip, left_shoulder, left_elbow)
            angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
            angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
            angle_right_elbow = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angle_right_shoulder = calculate_angle(right_hip, right_shoulder, right_elbow)
            angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)
            angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
            dist_y_l_wrist_shoulder = abs(left_wrist[1] - left_shoulder[1])
            dist_y_r_wrist_shoulder = abs(right_wrist[1] - right_shoulder[1])
            dist_z_l_wrist_hip = abs(left_wrist[2] - left_hip[2])
            dist_z_r_wrist_hip = abs(right_wrist[2] - right_hip[2])

            final_row_values = [ # Renamed to avoid confusion with DataFrame
                angle_left_elbow, angle_left_shoulder, angle_left_hip, angle_left_knee,
                angle_right_elbow, angle_right_shoulder, angle_right_hip, angle_right_knee,
                dist_y_l_wrist_shoulder, dist_y_r_wrist_shoulder,
                dist_z_l_wrist_hip, dist_z_r_wrist_hip
            ]

            # --- CRITICAL CHANGE HERE: Create DataFrame with correct feature names ---
            # Ensure 'feature_names' is loaded before this point
            X_live = pd.DataFrame([final_row_values], columns=feature_names)
            X_scaled = scaler.transform(X_live) # Transform the DataFrame
            # --- END CRITICAL CHANGE ---

            predicted_class = model.predict(X_scaled)[0]
            confidence = np.max(model.predict_proba(X_scaled))

            print(f"Predicted: {predicted_class:<30} | Confidence: {confidence:.2f}")
            if confidence >= CONF_THRESHOLD:
                prediction_buffer.append(predicted_class)
                if len(prediction_buffer) > STABILITY_FRAMES:
                    prediction_buffer.pop(0)

                if len(prediction_buffer) == STABILITY_FRAMES and len(set(prediction_buffer)) == 1:
                    new_stable_exercise = prediction_buffer[0]
                    if new_stable_exercise != stable_exercise:
                        stable_exercise = new_stable_exercise
                        rep_counter = 0
                        exercise_stage = None
            else:
                if len(prediction_buffer) != STABILITY_FRAMES or len(set(prediction_buffer)) != 1:
                    stable_exercise = "UNKNOWN"

            form_status = "CORRECT FORM"
            status_color = (0, 255, 0)

            # --- REP COUNTER AND FORM CORRECTION LOGIC ---
            # Squat/Lunge
            if 'squat' in stable_exercise or 'lunge' in stable_exercise:
                if angle_left_knee > 160: exercise_stage = "up"
                if angle_left_knee < 90 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1
                if 'squat' in stable_exercise and angle_left_hip < 70:
                    form_status = "GO DEEPER"; status_color = (0, 165, 255)

            # Deadlift/Bent Over Row/One Arm Dumbbell Row
            elif 'deadlift' in stable_exercise or 'bent_over' in stable_exercise or 'one_arm_dumbbell_row' in stable_exercise:
                back_angle = calculate_angle(left_shoulder, left_hip, left_knee) # Consider angle between torso and thigh
                if back_angle < 160: # If back is too horizontal
                    form_status = "KEEP BACK STRAIGHT"; status_color = (0, 0, 255)
                if 'deadlift' in stable_exercise:
                    # Deadlift stage based on hip extension
                    if angle_left_hip > 160: exercise_stage = "up"
                    if angle_left_hip < 90 and exercise_stage == 'up':
                        exercise_stage = "down"; rep_counter += 1
                else: # Bent Over Row / One Arm Dumbbell Row
                    # Row stage based on elbow flexion
                    if angle_left_elbow > 160: exercise_stage = "down"
                    if angle_left_elbow < 90 and exercise_stage == 'down': # Adjust angle as needed for full contraction
                        exercise_stage = "up"; rep_counter += 1

            # Dumbbell Pull Over
            elif 'pull_over' in stable_exercise:
                if angle_left_shoulder < 100: exercise_stage = "up"
                if angle_left_shoulder > 120 and exercise_stage == 'up': # Adjust angle as needed for full stretch
                    exercise_stage = "down"; rep_counter += 1

            # Bench Press/Push Up
            elif 'bench_press' in stable_exercise or 'push_up' in stable_exercise:
                if angle_left_elbow > 160: exercise_stage = "up"
                if angle_left_elbow < 90 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1

            # Chest Fly
            elif 'chest_fly' in stable_exercise:
                fly_angle = calculate_angle(right_shoulder, left_shoulder, left_elbow) # Angle between shoulders and elbow
                if fly_angle > 160: exercise_stage = "open"
                if fly_angle < 40 and exercise_stage == 'open': # Adjust angle for full contraction
                    exercise_stage = "closed"; rep_counter += 1

            # Shoulder Press/Overhead Press
            elif 'shoulder_press' in stable_exercise or 'overhead_press' in stable_exercise:
                if angle_left_elbow < 90: exercise_stage = "down"
                if angle_left_elbow > 160 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1

            # Lateral Raise/Front Raise/Rear Delt Fly
            elif 'lateral_raise' in stable_exercise or 'front_raise' in stable_exercise or 'rear_delt_fly' in stable_exercise:
                if angle_left_shoulder < 20: exercise_stage = "down"
                if angle_left_shoulder > 80 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1
                if angle_left_elbow < 150: # Check for bent elbow
                    form_status = "KEEP ARMS STRAIGHTER"; status_color = (0, 165, 255)

            # Upright Row
            elif 'upright_row' in stable_exercise:
                # Using Y-coordinate comparison for wrist position relative to shoulder
                if left_wrist[1] > left_shoulder[1]: exercise_stage = "down" # Wrist below shoulder (start)
                if left_wrist[1] < left_shoulder[1] and exercise_stage == 'down': # Wrist above shoulder (peak)
                    exercise_stage = "up"; rep_counter += 1

            # Bicep Curl/Straight Bar Curl
            elif 'bicep_curl' in stable_exercise or 'straight_bar_curl' in stable_exercise:
                if angle_left_elbow > 160: exercise_stage = "down" # Arm extended
                if angle_left_elbow < 30 and exercise_stage == 'down': # Arm fully curled
                    exercise_stage = "up"; rep_counter += 1

            # Tricep Extension/Skull Crusher/Tricep Kickback
            elif 'tricep_extension' in stable_exercise or 'skull_crusher' in stable_exercise or 'tricep_kickback' in stable_exercise:
                if angle_left_elbow < 90: exercise_stage = "up" # Arm extended (triceps contracted)
                if angle_left_elbow > 160 and exercise_stage == 'up': # Arm fully bent (triceps stretched)
                    exercise_stage = "down"; rep_counter += 1


        except (IndexError, TypeError) as e:
            # print(f"Landmark processing error: {e}") # Uncomment for debugging
            pass
        except Exception as e:
            # print(f"An unexpected error occurred during feature extraction or prediction: {e}") # Uncomment for debugging
            pass

    # Draw landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Connections
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Landmarks
                                 )

    # UI elements
    cv2.rectangle(image, (0, 0), (450, 110), UI_COLOR, -1)

    cv2.putText(image, 'EXERCISE', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    display_exercise = stable_exercise.replace('_', ' ').title()
    cv2.putText(image, display_exercise, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, 'REPS', (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, str(rep_counter), (295, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(image, (0, image.shape[0] - 60), (image.shape[1], image.shape[0]), status_color, -1)
    cv2.putText(image, form_status, (15, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
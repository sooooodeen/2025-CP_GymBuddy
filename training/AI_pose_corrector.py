import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- Constants ---
MODEL_FILENAME = 'exercise_model_engineered.pkl'
CONF_THRESHOLD = 0.80  # Only consider predictions with >= 80% confidence
STABILITY_FRAMES = 15  # Lock in prediction after this many consecutive frames
UI_COLOR = (245, 117, 16) # BGR color for the UI boxes

# --- Helper Function to Calculate Angles ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Load the Trained Model ---
try:
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_FILENAME}'")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Create a named window ---
window_name = 'AI Fitness Trainer'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1600, 900)

# --- State and Counter Variables ---
rep_counter = 0
exercise_stage = None
form_status = "CORRECT FORM"
status_color = (0, 255, 0) # Green for correct

# --- Stability Buffer ---
prediction_buffer = []
stable_exercise = "UNKNOWN"

print("--- Live feed started. Press 'q' to quit. ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Process the frame for pose detection
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- Prediction and Feature Logic ---
    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- 1. PREDICT EXERCISE (with stability) ---
            # Calculate engineered features in real-time (must match training script)
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]

            # Feature calculation
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

            # Assemble the final feature vector IN THE CORRECT ORDER
            final_row = [
                angle_left_elbow, angle_left_shoulder, angle_left_hip, angle_left_knee,
                angle_right_elbow, angle_right_shoulder, angle_right_hip, angle_right_knee,
                dist_y_l_wrist_shoulder, dist_y_r_wrist_shoulder,
                dist_z_l_wrist_hip, dist_z_r_wrist_hip
            ]
            
            # Make prediction
            X = np.array([final_row])
            predicted_class = model.predict(X)[0]
            confidence = np.max(model.predict_proba(X))

            # --- STABILITY BUFFER LOGIC ---
            if confidence >= CONF_THRESHOLD:
                prediction_buffer.append(predicted_class)
                if len(prediction_buffer) > STABILITY_FRAMES:
                    prediction_buffer.pop(0) # Keep buffer size constant
                
                # Check if the buffer is full and all predictions are the same
                if len(prediction_buffer) == STABILITY_FRAMES and len(set(prediction_buffer)) == 1:
                    new_stable_exercise = prediction_buffer[0]
                    if new_stable_exercise != stable_exercise:
                        stable_exercise = new_stable_exercise
                        rep_counter = 0 # Reset counter for new exercise
                        exercise_stage = None
            else:
                # If confidence is low, reset buffer and go to UNKNOWN state
                prediction_buffer = []
                stable_exercise = "UNKNOWN"

            # --- 2. REP COUNTING & FORM CHECKING (operates on STABLE prediction) ---
            form_status = "CORRECT FORM"
            status_color = (0, 255, 0) # Green
            
            # This logic only runs if an exercise has been stably identified
            if 'squat' in stable_exercise or 'lunge' in stable_exercise:
                if angle_left_knee > 160: exercise_stage = "up"
                if angle_left_knee < 90 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1
                if 'squat' in stable_exercise and angle_left_hip < 70: # More specific squat form check
                    form_status = "GO DEEPER"; status_color = (0, 165, 255) # Orange

            elif 'deadlift' in stable_exercise or 'bent_over' in stable_exercise or 'one_arm_dumbbell_row' in stable_exercise:
                back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                if back_angle < 160:
                    form_status = "KEEP BACK STRAIGHT"; status_color = (0, 0, 255) # Red
                if 'deadlift' in stable_exercise:
                    if angle_left_hip > 160: exercise_stage = "up"
                    if angle_left_hip < 90 and exercise_stage == 'up':
                        exercise_stage = "down"; rep_counter += 1
                else: # For rows
                    if angle_left_elbow > 160: exercise_stage = "down"
                    if angle_left_elbow < 90 and exercise_stage == 'down':
                        exercise_stage = "up"; rep_counter += 1
            
            elif 'pull-over' in stable_exercise:
                if angle_left_shoulder < 100: exercise_stage = "up"
                if angle_left_shoulder > 120 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1

            elif 'bench_press' in stable_exercise or 'push_up' in stable_exercise:
                if angle_left_elbow > 160: exercise_stage = "up"
                if angle_left_elbow < 90 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1

            elif 'chest_fly' in stable_exercise:
                fly_angle = calculate_angle(right_shoulder, left_shoulder, left_elbow)
                if fly_angle > 160: exercise_stage = "open"
                if fly_angle < 40 and exercise_stage == 'open':
                    exercise_stage = "closed"; rep_counter += 1

            elif 'shoulder_press' in stable_exercise or 'overhead_press' in stable_exercise:
                if angle_left_elbow < 90: exercise_stage = "down"
                if angle_left_elbow > 160 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1
            
            elif 'lateral_raise' in stable_exercise or 'front_raise' in stable_exercise or 'rear_delt_fly' in stable_exercise:
                if angle_left_shoulder < 20: exercise_stage = "down"
                if angle_left_shoulder > 80 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1
                if angle_left_elbow < 150: # Check for bent elbows
                    form_status = "KEEP ARMS STRAIGHTER"; status_color = (0, 165, 255) # Orange

            elif 'upright_row' in stable_exercise:
                if left_wrist[1] > left_shoulder[1]: exercise_stage = "down"
                if left_wrist[1] < left_shoulder[1] and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1

            elif 'bicep_curl' in stable_exercise or 'straight_bar_curl' in stable_exercise:
                if angle_left_elbow > 160: exercise_stage = "down"
                if angle_left_elbow < 30 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1
            
            elif 'tricep_extension' in stable_exercise or 'skull_crusher' in stable_exercise or 'tricep_kickback' in stable_exercise:
                if angle_left_elbow < 90: exercise_stage = "up"
                if angle_left_elbow > 160 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1
            
        # IMPROVED ERROR HANDLING: Print the error but don't crash
        except (IndexError, TypeError) as e:
            print(f"Error processing landmarks: {e}. A body part might be out of frame.")
            # We can let the loop continue, it will just miss this frame.
            pass
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            pass

    # --- 3. RENDER RESULTS ---
    # Draw landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Status Box
    cv2.rectangle(image, (0, 0), (450, 110), UI_COLOR, -1)
    
    # Class Name
    cv2.putText(image, 'EXERCISE', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    display_exercise = stable_exercise.replace('_', ' ').title()
    cv2.putText(image, display_exercise, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Reps
    cv2.putText(image, 'REPS', (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, str(rep_counter), (295, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Form Status Bar
    cv2.rectangle(image, (0, image.shape[0] - 60), (image.shape[1], image.shape[0]), status_color, -1)
    cv2.putText(image, form_status, (15, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
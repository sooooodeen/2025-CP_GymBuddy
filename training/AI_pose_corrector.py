import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- Helper Function to Calculate Angles ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Load the Trained Model ---
# IMPORTANT: This now points to your new, smarter model
model_filename = 'exercise_model_engineered.pkl' 
try:
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_filename}'")
    exit()

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
# Request 1920x1080 resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Create a named window and set it to fullscreen ---
window_name = 'AI Fitness Trainer'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# --- Variables for Features ---
rep_counter = 0
exercise_stage = None
current_exercise = ""
form_status = ""
status_color = (0, 255, 0) # Default to green (BGR color format)

print("--- Live feed started. Press 'q' to quit. ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Process the frame for pose detection
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- Prediction and Feature Logic ---
    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            
            # 1. PREDICT EXERCISE
            # Extract raw coordinates first
            raw_coords_row = []
            for lm in landmarks:
                raw_coords_row.extend([lm.x, lm.y, lm.z, lm.visibility])

            # Calculate the same engineered features in real-time
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            angle_left_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)
            angle_left_shoulder = calculate_angle(left_hip, left_shoulder, left_elbow)
            angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
            angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
            dist_y_wrist_shoulder = abs(left_wrist[1] - left_shoulder[1])
            dist_y_elbow_shoulder = abs(left_elbow[1] - left_shoulder[1])

            # Combine raw coordinates with engineered features to create the final input row
            final_row = raw_coords_row + [
                angle_left_elbow, angle_left_shoulder, angle_left_hip, angle_left_knee,
                dist_y_wrist_shoulder, dist_y_elbow_shoulder
            ]
            
            # Make prediction with the complete feature set
            X = np.array([final_row])
            predicted_class = model.predict(X)[0]
            confidence = np.max(model.predict_proba(X))

            if predicted_class != current_exercise:
                rep_counter = 0
                exercise_stage = None
                current_exercise = predicted_class

            # 2. FEATURE LOGIC (Warning System)
            form_status = "CORRECT FORM"
            status_color = (0, 255, 0) # Green

            # --- Full Exercise Logic (uses the already calculated angles) ---
            if 'squat' in predicted_class or 'lunge' in predicted_class:
                if angle_left_knee > 160: exercise_stage = "up"
                if angle_left_knee < 90 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1
                if 'squat' in predicted_class and angle_left_hip > 90:
                    form_status = "INCORRECT FORM"; status_color = (0, 0, 255)

            elif 'deadlift' in predicted_class or 'bent_over' in predicted_class or 'one_arm_dumbbell_row' in predicted_class:
                back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                if back_angle < 160:
                    form_status = "INCORRECT FORM"; status_color = (0, 0, 255)
                if 'deadlift' in predicted_class:
                    if back_angle > 160: exercise_stage = "up"
                    if back_angle < 90 and exercise_stage == 'up':
                        exercise_stage = "down"; rep_counter += 1
                else: # For rows
                    if angle_left_elbow > 160: exercise_stage = "down"
                    if angle_left_elbow < 90 and exercise_stage == 'down':
                        exercise_stage = "up"; rep_counter += 1
            
            elif 'pull-over' in predicted_class:
                if angle_left_shoulder < 100: exercise_stage = "up"
                if angle_left_shoulder > 120 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1

            elif 'bench_press' in predicted_class or 'push_up' in predicted_class:
                if angle_left_elbow > 160: exercise_stage = "up"
                if angle_left_elbow < 90 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1

            elif 'chest_fly' in predicted_class:
                fly_angle = calculate_angle(right_shoulder, left_shoulder, left_elbow)
                if fly_angle > 160: exercise_stage = "open"
                if fly_angle < 40 and exercise_stage == 'open':
                    exercise_stage = "closed"; rep_counter += 1

            elif 'shoulder_press' in predicted_class or 'overhead_press' in predicted_class:
                if angle_left_elbow < 90: exercise_stage = "down"
                if angle_left_elbow > 160 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1
            
            elif 'lateral_raise' in predicted_class or 'front_raise' in predicted_class or 'rear_delt_fly' in predicted_class:
                if angle_left_shoulder < 20: exercise_stage = "down"
                if angle_left_shoulder > 80 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1

            elif 'upright_row' in predicted_class:
                if left_wrist[1] > left_shoulder[1]: exercise_stage = "down" # wrist_y > shoulder_y
                if left_wrist[1] < left_shoulder[1] and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1

            elif 'bicep_curl' in predicted_class or 'straight_bar_curl' in predicted_class:
                if angle_left_elbow > 160: exercise_stage = "down"
                if angle_left_elbow < 30 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1
            
            elif 'tricep_extension' in predicted_class or 'skull_crusher' in predicted_class or 'tricep_kickback' in predicted_class:
                if angle_left_elbow < 90: exercise_stage = "up"
                if angle_left_elbow > 160 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1
            
            
            # 3. RENDER RESULTS
            cv2.rectangle(image, (0, 0), (400, 110), (245, 117, 16), -1)
            cv2.putText(image, 'CLASS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, current_exercise.split('_')[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'{confidence:.2f}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'REPS', (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(rep_counter), (245, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (0, image.shape[0] - 60), (image.shape[1], image.shape[0]), status_color, -1)
            cv2.putText(image, form_status, (15, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            pass

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


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
model_filename = 'exercise_model.pkl'
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
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Variables for Features ---
rep_counter = 0
exercise_stage = None  # Can be 'up' or 'down', 'in' or 'out', etc.
feedback = ""
current_exercise = ""

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
            row = []
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            X = np.array([row])
            predicted_class = model.predict(X)[0]
            confidence = np.max(model.predict_proba(X))

            # --- Reset counter if exercise changes ---
            if predicted_class != current_exercise:
                rep_counter = 0
                exercise_stage = None
                current_exercise = predicted_class

            # 2. FEATURE LOGIC (Rep Counting and Form Correction)
            feedback = ""  # Reset feedback each frame
            
            # For simplicity, logic defaults to the LEFT side of the body.

            # --- Legs & Glutes ---
            if 'squat' in predicted_class or 'lunge' in predicted_class:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)
                
                if knee_angle > 160:
                    exercise_stage = "up"
                if knee_angle < 90 and exercise_stage == 'up':
                    exercise_stage = "down"
                    rep_counter += 1
                
                if 'squat' in predicted_class:
                    if hip_angle < 80:
                        feedback = "Great Depth!"
                    else:
                        feedback = "Go Deeper!"

            # --- Back ---
            elif 'deadlift' in predicted_class or 'bent_over' in predicted_class or 'one_arm_dumbbell_row' in predicted_class:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                back_angle = calculate_angle(shoulder, hip, knee)
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                if back_angle < 160:
                    feedback = "Keep Your Back Straight"
                
                if 'deadlift' in predicted_class:
                    if back_angle > 160: exercise_stage = "up"
                    if back_angle < 90 and exercise_stage == 'up':
                        exercise_stage = "down"; rep_counter += 1
                else: # For rows
                    if elbow_angle > 160: exercise_stage = "down"
                    if elbow_angle < 90 and exercise_stage == 'down':
                        exercise_stage = "up"; rep_counter += 1
            
            elif 'pull-over' in predicted_class:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                arm_angle = calculate_angle(hip, shoulder, elbow)
                if arm_angle < 100: exercise_stage = "up"
                if arm_angle > 120 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1

            # --- Chest ---
            elif 'bench_press' in predicted_class or 'push_up' in predicted_class:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                if elbow_angle > 160: exercise_stage = "up"
                if elbow_angle < 90 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1

            elif 'chest_fly' in predicted_class:
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                fly_angle = calculate_angle(right_shoulder, left_shoulder, left_elbow)
                if fly_angle > 160: exercise_stage = "open"
                if fly_angle < 40 and exercise_stage == 'open':
                    exercise_stage = "closed"; rep_counter += 1

            # --- Shoulders ---
            elif 'shoulder_press' in predicted_class or 'overhead_press' in predicted_class:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                if elbow_angle < 90: exercise_stage = "down"
                if elbow_angle > 160 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1
            
            elif 'lateral_raise' in predicted_class or 'front_raise' in predicted_class or 'rear_delt_fly' in predicted_class:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                arm_angle = calculate_angle(hip, shoulder, elbow)
                if arm_angle < 20: exercise_stage = "down"
                if arm_angle > 80 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1

            elif 'upright_row' in predicted_class:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                if wrist_y > shoulder_y: exercise_stage = "down"
                if wrist_y < shoulder_y and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1

            # --- Arms ---
            elif 'bicep_curl' in predicted_class or 'straight_bar_curl' in predicted_class:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                if elbow_angle > 160: exercise_stage = "down"
                if elbow_angle < 30 and exercise_stage == 'down':
                    exercise_stage = "up"; rep_counter += 1
            
            elif 'tricep_extension' in predicted_class or 'skull_crusher' in predicted_class or 'tricep_kickback' in predicted_class:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                if elbow_angle < 90: exercise_stage = "up"
                if elbow_angle > 160 and exercise_stage == 'up':
                    exercise_stage = "down"; rep_counter += 1

            # 3. RENDER RESULTS
            # Status Box
            cv2.rectangle(image, (0, 0), (400, 110), (245, 117, 16), -1)

            # Display Class and Confidence
            cv2.putText(image, 'CLASS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, current_exercise.split('_')[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'{confidence:.2f}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Rep Counter
            cv2.putText(image, 'REPS', (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(rep_counter), (245, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Feedback
            if feedback:
                cv2.rectangle(image, (0, 420), (image.shape[1], 480), (245, 117, 16), -1)
                cv2.putText(image, feedback, (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            pass

    # Draw the skeleton
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    cv2.imshow('AI Fitness Trainer', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


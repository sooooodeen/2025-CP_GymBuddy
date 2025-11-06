import numpy as np
import mediapipe as mp
import time
import tensorflow as tf # Added for TFLite object types
from collections import deque, Counter

# --- 3D ANGLE FUNCTION (For Model Prediction) ---
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

# --- 2D ANGLE FUNCTION (For Form Checking) ---
def calculate_angle_2d(a, b, c):
    """This 2D version is for the form checker, which uses 2D coordinate arrays."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- FEATURE EXTRACTION (For Model Prediction) ---
def extract_angle_features_for_model(landmarks):
    """Extracts the 8 key angles using the full 3D landmark data."""
    lm = landmarks
    mp_pose = mp.solutions.pose.PoseLandmark

    # Ensure all landmarks are present before calculating
    try:
        return np.array([
            calculate_angle(lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_ELBOW], lm[mp_pose.LEFT_WRIST]),
            calculate_angle(lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_ELBOW], lm[mp_pose.RIGHT_WRIST]),
            calculate_angle(lm[mp_pose.LEFT_ELBOW], lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_HIP]),
            calculate_angle(lm[mp_pose.RIGHT_ELBOW], lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_HIP]),
            calculate_angle(lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_HIP], lm[mp_pose.LEFT_KNEE]),
            calculate_angle(lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_HIP], lm[mp_pose.RIGHT_KNEE]),
            calculate_angle(lm[mp_pose.LEFT_HIP], lm[mp_pose.LEFT_KNEE], lm[mp_pose.LEFT_ANKLE]),
            calculate_angle(lm[mp_pose.RIGHT_HIP], lm[mp_pose.RIGHT_KNEE], lm[mp_pose.RIGHT_ANKLE]),
            calculate_angle(lm[mp_pose.LEFT_HIP], lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_ELBOW]),
            calculate_angle(lm[mp_pose.RIGHT_HIP], lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_ELBOW])
        ])
    except Exception as e:
        # Return None or NaNs if any landmark is missing
        print(f"Error extracting features: {e}")
        return None


# --- The New Advanced Exercise Analysis Class ---
class ExerciseAnalyzer:
    def __init__(self, sequence_length=90, conf_threshold=0.80, stability_frames=10, reset_timeout=5.0):
        self.rep_counter = 0
        self.stage = None
        self.form_status = "START EXERCISE"
        self.status_color = (0, 255, 0)
        self.previous_exercise = "neutral"
        self.last_rep_time = time.time()
        self.RESET_TIMEOUT = reset_timeout
        self.debug_angles = {} # For the debug view
        self.angle_sequence_buffer = deque(maxlen=sequence_length)
        self.SEQUENCE_LENGTH = sequence_length
        self.CONF_THRESHOLD = conf_threshold
        self.STABILITY_FRAMES = stability_frames
        self.recent_predictions = deque(maxlen=stability_frames)
        self.stable_prediction = "Neutral" # Added for prediction output
        self.frame_count = 0 # Added to control prediction interval
        self.PREDICTION_INTERVAL = 3 # Run prediction every 3 frames
        self.triggered_alert = None
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.new_error_to_log = None    

    def predict_with_tflite(self, interpreter, input_details, output_details, feature_sequence):
        """
        Runs inference on the TFLite interpreter. 
        Uses Float32 I/O since the model was converted with Weight-Only Quantization.
        """
        
        # Reshape for TFLite input (1, SEQUENCE_LENGTH, FEATURE_SIZE)
        input_data = np.expand_dims(feature_sequence, axis=0).astype(np.float32)

        # 1. Set the input tensor
        # Input tensor is always at index 0
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # 2. Run the inference (This is the fast part!)
        interpreter.invoke()
        
        # 3. Get the output tensor (raw prediction probabilities)
        output = interpreter.get_tensor(output_details[0]['index'])
        
        return output[0] # Return the prediction array (e.g., [0.1, 0.9, 0.0])
    
    def get_triggered_alert(self):
        """ Checks if a trainer alert was triggered and clears it. """
        alert_to_send = self.triggered_alert
        self.triggered_alert = None # Clear the alert after fetching
        return alert_to_send
    
    def get_new_error_log(self):
        """Called by app.py to get a new error log and clear it."""
        log_entry = self.new_error_to_log
        self.new_error_to_log = None # Clear after fetching
        return log_entry
    
    def process_frame(self, interpreter, input_details, output_details, label_mapping, landmarks, current_exercise):
        """
        Handles feature extraction, model prediction (via TFLite), rep counting, 
        and form logic for a single frame.
        """
        self.frame_count += 1
        
        # 1. Feature Extraction
        features = extract_angle_features_for_model(landmarks)
        if features is None:
            self.form_status = "NO PERSON DETECTED"
            return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

        # 2. Update the internal sequence buffer for the LSTM
        self.angle_sequence_buffer.append(features)
        
        # 3. Model Prediction (TFLite)
        if len(self.angle_sequence_buffer) == self.SEQUENCE_LENGTH and self.frame_count % self.PREDICTION_INTERVAL == 0:
            
            prediction_output = self.predict_with_tflite(
                interpreter, 
                input_details, 
                output_details, 
                np.array(self.angle_sequence_buffer)
            )

            print(f"Raw Prediction Output: {prediction_output}")
            
            # Post-processing (stability logic)
            predicted_label_index = np.argmax(prediction_output)
            confidence = prediction_output[predicted_label_index]

            print(f"Predicted Index: {predicted_label_index}, Confidence: {confidence: .2f}")
            
            if confidence > self.CONF_THRESHOLD:
                predicted_label = label_mapping.get(int(predicted_label_index), "Unknown")
                self.recent_predictions.append(predicted_label)
                print(f"Label Mapping Result: {predicted_label}")
            else:
                self.recent_predictions.append("Neutral") # Use neutral if confidence is low
                print("Confidence below threshold, adding Neutral.")

            # Check for stable prediction over the queue
            prediction_counts = Counter(self.recent_predictions)
            most_common, count = prediction_counts.most_common(1)[0]

            print(f"Recent Predictions Buffer: {list(self.recent_predictions)}")
            print(f"Prediction Counts: {prediction_counts}")
            print(f"Most Common: {most_common}, Count: {count}")
            
            if count >= self.STABILITY_FRAMES:
                # Check if the stable prediction is *changing*
                if self.stable_prediction != most_common:
                    print(f"!!! Stable Prediction CHANGED: {most_common} !!!")
                    self.stable_prediction = most_common
                    
                    # CRITICAL: Clear all buffers to force a "fresh look"
                    # This stops the model from getting confused by old data.
                    self.angle_sequence_buffer.clear()
                    self.recent_predictions.clear()
            
        # 4. Run Rule-Based Form Analysis and Rep Counting
        # The form check depends on the stable prediction for the current exercise
        self.analyze_frame(self.stable_prediction, landmarks)

        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if landmarks is None:
            if exercise_name == "neutral":
                self.previous_exercise = "neutral"
                self.stage = None
            self.debug_angles.clear()
            return

        # Reset counters if exercise changes
        if exercise_name != self.previous_exercise:
            self.rep_counter = 0
            self.stage = None
            self.previous_exercise = exercise_name
            self.last_rep_time = time.time()

        self.form_status = "CORRECT FORM"
        self.status_color = (0, 255, 0)

        # Reset stage if user is inactive
        if time.time() - self.last_rep_time > self.RESET_TIMEOUT and self.stage is not None:
            self.stage = None
            self.form_status = "INACTIVE - RESET"
        
        mp_lm = mp.solutions.pose.PoseLandmark
        
        # Wrap all exercise logic in a try-except for robustness
        try:
            prev_rep_counter = self.rep_counter

            if 'bicepCurl' in exercise_name:
                left_shoulder_lm = landmarks[mp_lm.LEFT_SHOULDER.value]; left_elbow_lm = landmarks[mp_lm.LEFT_ELBOW.value]; left_wrist_lm = landmarks[mp_lm.LEFT_WRIST.value]; left_hip_lm = landmarks[mp_lm.LEFT_HIP.value]
                right_shoulder_lm = landmarks[mp_lm.RIGHT_SHOULDER.value]; right_elbow_lm = landmarks[mp_lm.RIGHT_ELBOW.value]; right_wrist_lm = landmarks[mp_lm.RIGHT_WRIST.value]; right_hip_lm = landmarks[mp_lm.RIGHT_HIP.value]
                
                elbow_angle, shoulder_angle = 0, 0
                # Choose the most visible arm for analysis
                if left_wrist_lm.visibility > right_wrist_lm.visibility:
                    elbow_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_elbow_lm.x, left_elbow_lm.y], [left_wrist_lm.x, left_wrist_lm.y])
                    shoulder_angle = calculate_angle_2d([left_elbow_lm.x, left_elbow_lm.y], [left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y])
                else:
                    elbow_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_elbow_lm.x, right_elbow_lm.y], [right_wrist_lm.x, right_wrist_lm.y])
                    shoulder_angle = calculate_angle_2d([right_elbow_lm.x, right_elbow_lm.y], [right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y])
                self.debug_angles = {'Elbow': elbow_angle, 'Shoulder': shoulder_angle}

                # Rep counting logic
                if elbow_angle > 150: self.stage = "down"
                if elbow_angle < 45 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                # Form correction
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
                    left_torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
                    right_torso_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y], [right_knee_lm.x, right_knee_lm.y])
                    torso_angle = left_torso_angle if left_hip_lm.visibility > right_hip_lm.visibility else right_torso_angle
                    self.debug_angles = {'Elbow': elbow_angle, 'Torso': torso_angle}
                    
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
                    left_torso_angle = calculate_angle_2d([left_shoulder_lm.x, left_shoulder_lm.y], [left_hip_lm.x, left_hip_lm.y], [left_knee_lm.x, left_knee_lm.y])
                    right_torso_angle = calculate_angle_2d([right_shoulder_lm.x, right_shoulder_lm.y], [right_hip_lm.x, right_hip_lm.y], [right_knee_lm.x, right_knee_lm.y])
                    torso_angle = left_torso_angle if left_hip_lm.visibility > right_hip_lm.visibility else right_torso_angle
                    self.debug_angles = {'Elbow': elbow_angle, 'Torso': torso_angle}
                    
                    if torso_angle > 135:
                        self.form_status = "ERROR: BEND OVER MORE (45 DEG)"; self.status_color = (0, 0, 255)

            is_new_rep = (self.rep_counter > prev_rep_counter)

            if is_new_rep:
                # 1. Check for immediate error logging
                if "ERROR" in self.form_status:
                    # Set the new_error_to_log variable. app.py will read this.
                    self.new_error_to_log = { 
                        "rep_number": self.rep_counter, 
                        "error_type": self.form_status, 
                        "exercise_name": exercise_name 
                    }
                
                # 2. Check for trainer alert (consecutive errors)
                if "ERROR" in self.form_status:
                    if self.form_status == self.last_consecutive_error_type:
                        self.consecutive_error_counter += 1
                    else:
                        self.last_consecutive_error_type = self.form_status
                        self.consecutive_error_counter = 1
                else: # Correct rep resets the counter
                    self.consecutive_error_counter = 0
                    self.last_consecutive_error_type = None

                if self.consecutive_error_counter >= 6:
                    alert_message = f"Repeated Error: {self.last_consecutive_error_type.replace('ERROR: ', '')}"
                    self.triggered_alert = {'message': alert_message, 'exercise': exercise_name, 'reps': self.rep_counter}
                    print(f"!!! TRAINER ALERT TRIGGERED: {alert_message} !!!")
                    self.consecutive_error_counter = 0

        except Exception as e:
            print(f"Error during form analysis for {exercise_name}: {e}")
            self.form_status = "ERROR: TRACKING LOST"
            self.status_color = (0,0,255)
            self.debug_angles.clear()
            pass
    
    def reset_session(self):
        """Resets all counters for a new session."""
        self.rep_counter = 0
        self.stage = None
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.triggered_alert = None
        self.new_error_to_log = None
        print("Analyzer reset for new session.")

    def get_status(self):
        return self.rep_counter, self.form_status, self.status_color, self.debug_angles


import numpy as np
import mediapipe as mp
import time
import tensorflow as tf 
from collections import deque, Counter

# --- 3D ANGLE FUNCTION (Kept for consistency) ---
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

# --- FEATURE EXTRACTION (UPDATED: Raw Landmarks for AI Model) ---
def extract_angle_features_for_model(landmarks):
    """
    Extracts flattened 132 raw values (x, y, z, visibility) to match
    the trained model's input requirement.
    """
    try:
        row = []
        for lm in landmarks:
            # Extract x, y, z, and visibility for every landmark
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # Return as a float32 numpy array
        return np.array(row, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# --- Advanced Exercise Analysis Class ---
class ExerciseAnalyzer:
    def __init__(self, sequence_length=90, conf_threshold=0.50, stability_frames=5, reset_timeout=5.0):
        self.rep_counter = 0
        self.stage = None
        self.form_status = "START EXERCISE"
        self.status_color = (0, 255, 0)
        self.previous_exercise = "neutral"
        self.last_rep_time = time.time()
        self.RESET_TIMEOUT = reset_timeout
        self.debug_angles = {} 
        
        # --- Sequence Buffer ---
        self.SEQUENCE_LENGTH = sequence_length
        self.angle_sequence_buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        
        # --- Prediction Stability ---
        self.CONF_THRESHOLD = conf_threshold
        self.STABILITY_FRAMES = stability_frames
        self.recent_predictions = deque(maxlen=self.STABILITY_FRAMES)
        self.stable_prediction = "neutral"
        self.frame_count = 0 
        self.PREDICTION_INTERVAL = 3 
        
        # --- Error Logging ---
        self.triggered_alert = None
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.new_error_to_log = None    

    def predict_with_tflite(self, interpreter, input_details, output_details, feature_sequence):
        """
        Robust Inference: Automatically handles Quantization (Float -> Int8).
        """
        input_index = input_details[0]['index']
        input_dtype = input_details[0]['dtype']
        
        # 1. Prepare Input Data (Add Batch Dimension)
        # Expected Shape: (1, 90, 132)
        input_data = np.expand_dims(feature_sequence, axis=0)

        # 2. CHECK FOR QUANTIZATION (The Fix)
        if input_dtype != np.float32:
            # If the model expects integers (int8/uint8), we must quantize our floats
            scale, zero_point = input_details[0]['quantization']
            
            if scale > 0:
                # Formula: q = r / S + Z
                input_data = (input_data / scale) + zero_point
                # Clip to ensure valid range for int8 (-128, 127) or uint8 (0, 255)
                if input_dtype == np.int8:
                    input_data = np.clip(input_data, -128, 127)
                else:
                    input_data = np.clip(input_data, 0, 255)
                
                input_data = input_data.astype(input_dtype)

        # 3. Run Inference
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        
        # 4. Process Output
        output_index = output_details[0]['index']
        output_data = interpreter.get_tensor(output_index)[0]
        
        # De-quantize output if necessary (Int8 -> Float)
        output_dtype = output_details[0]['dtype']
        if output_dtype != np.float32:
            scale, zero_point = output_details[0]['quantization']
            if scale > 0:
                output_data = (output_data.astype(np.float32) - zero_point) * scale

        return output_data
    
    def get_triggered_alert(self):
        alert_to_send = self.triggered_alert
        self.triggered_alert = None 
        return alert_to_send
    
    def get_new_error_log(self):
        log_entry = self.new_error_to_log
        self.new_error_to_log = None 
        return log_entry
    
    def process_frame(self, interpreter, input_details, output_details, label_mapping, landmarks, current_exercise):
        self.frame_count += 1
        
        # 1. Feature Extraction
        features = extract_angle_features_for_model(landmarks)
        if features is None:
            self.form_status = "NO PERSON DETECTED"
            return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

        # 2. Update Buffer
        self.angle_sequence_buffer.append(features)
        
        # 3. Model Prediction
        if len(self.angle_sequence_buffer) == self.SEQUENCE_LENGTH and self.frame_count % self.PREDICTION_INTERVAL == 0:
            try:
                prediction_output = self.predict_with_tflite(
                    interpreter, input_details, output_details, np.array(self.angle_sequence_buffer)
                )

                predicted_idx = np.argmax(prediction_output)
                confidence = prediction_output[predicted_idx]
                
                # Optional: Softmax if output is not normalized
                if np.sum(prediction_output) > 1.1: 
                    exp_x = np.exp(prediction_output - np.max(prediction_output))
                    prediction_output = exp_x / exp_x.sum()
                    confidence = prediction_output[predicted_idx]

                # Debug print (view in your terminal to verify it's working)
                # print(f"Pred: {predicted_idx}, Conf: {confidence:.2f}")

                if confidence > self.CONF_THRESHOLD:
                    predicted_label = label_mapping.get(int(predicted_idx), "Unknown")
                    self.recent_predictions.append(predicted_label)
                else:
                    self.recent_predictions.append("neutral")

                # Stability Check
                prediction_counts = Counter(self.recent_predictions)
                most_common, count = prediction_counts.most_common(1)[0]
                
                if count >= (self.STABILITY_FRAMES - 2):
                    if self.stable_prediction != most_common:
                        self.stable_prediction = most_common
                        # Optional: clear buffer on switch if needed, usually better to keep rolling
                        # self.angle_sequence_buffer.clear() 
                        self.recent_predictions.clear()
            
            except Exception as e:
                print(f"Prediction Error: {e}")
            
        # 4. Run Rule-Based Form Analysis
        self.analyze_frame(self.stable_prediction, landmarks)

        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if landmarks is None or exercise_name == "neutral":
            if exercise_name == "neutral":
                self.previous_exercise = "neutral"
                self.stage = None
            self.debug_angles.clear()
            return

        # Reset if exercise changes
        if exercise_name != self.previous_exercise:
            self.rep_counter = 0
            self.stage = None
            self.previous_exercise = exercise_name
            self.last_rep_time = time.time()

        self.form_status = "CORRECT FORM"
        self.status_color = (0, 255, 0)

        # Reset inactive users
        if time.time() - self.last_rep_time > self.RESET_TIMEOUT and self.stage is not None:
            self.stage = None
            self.form_status = "INACTIVE - RESET"
        
        try:
            prev_rep_counter = self.rep_counter
            
            # --- EXTRACT COMMON LANDMARKS ONCE ---
            ls = landmarks[11]; le = landmarks[13]; lw = landmarks[15] # Left: Shoulder, Elbow, Wrist
            rs = landmarks[12]; re = landmarks[14]; rw = landmarks[16] # Right: Shoulder, Elbow, Wrist
            lh = landmarks[23]; lk = landmarks[25]; la = landmarks[27] # Left: Hip, Knee, Ankle
            rh = landmarks[24]; rk = landmarks[26]; ra = landmarks[28] # Right: Hip, Knee, Ankle

            # --- EXERCISE LOGIC BLOCKS (PRESERVED FROM YOUR ORIGINAL CODE) ---

            # 1. BICEP CURL
            if exercise_name == 'bicepCurl':
                if lw.visibility > rw.visibility:
                    elbow_angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                    shoulder_angle = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                else:
                    elbow_angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                    shoulder_angle = calculate_angle_2d([re.x, re.y], [rs.x, rs.y], [rh.x, rh.y])
                
                self.debug_angles = {'Elbow': int(elbow_angle)}
                
                if elbow_angle > 160: self.stage = "down"
                if elbow_angle < 45 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                if shoulder_angle > 45:
                    self.form_status = "ERROR: ELBOWS SWINGING"; self.status_color = (0, 0, 255)

            # 2. TRICEP KICKBACK
            elif exercise_name == 'tricepKickback':
                if lw.visibility > rw.visibility:
                    elbow_angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                    torso_angle = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                else:
                    elbow_angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                    torso_angle = calculate_angle_2d([rs.x, rs.y], [rh.x, rh.y], [rk.x, rk.y])
                
                self.debug_angles = {'Elbow': int(elbow_angle)}
                
                if elbow_angle < 90: self.stage = "in"
                if elbow_angle > 160 and self.stage == 'in':
                    self.stage = "out"; self.rep_counter += 1; self.last_rep_time = time.time()
                if torso_angle > 135:
                    self.form_status = "ERROR: BEND OVER MORE"

            # 3. LATERAL RAISE & REVERSE FLY
            elif exercise_name in ['lateralRaise', 'dumbbellReverseFly']:
                if le.visibility > re.visibility:
                    arm_angle = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                else:
                    arm_angle = calculate_angle_2d([re.x, re.y], [rs.x, rs.y], [rh.x, rh.y])
                
                self.debug_angles = {'Shoulder': int(arm_angle)}

                if arm_angle < 30: self.stage = "down"
                if arm_angle > 80 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                if arm_angle > 100:
                    self.form_status = "WARNING: DO NOT OVER-RAISE"

            # 4. VERTICAL & INCLINE PRESSES (Shoulder, Push Press, Incline Bench)
            elif exercise_name in ['shoulderPress', 'dumbbellPushPress', 'inclineBenchPress']:
                l_elb = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                r_elb = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                avg_elb = (l_elb + r_elb) / 2
                
                self.debug_angles = {'Elbow': int(avg_elb)}

                if avg_elb < 80: self.stage = "down"
                if avg_elb > 150 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()

            # 5. HORIZONTAL PRESS (Svend Press)
            elif exercise_name == 'dumbbellSvendPress':
                l_elb = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                r_elb = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                avg_elb = (l_elb + r_elb) / 2
                
                self.debug_angles = {'Elbow': int(avg_elb)}

                if avg_elb < 90: self.stage = "in"
                if avg_elb > 160 and self.stage == 'in':
                    self.stage = "out"; self.rep_counter += 1; self.last_rep_time = time.time()

            # 6. UPRIGHT ROW
            elif exercise_name == 'uprightRow':
                l_elb = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                r_elb = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                avg_elb = (l_elb + r_elb) / 2
                
                self.debug_angles = {'Elbow': int(avg_elb)}

                if avg_elb > 140: self.stage = "down"
                if avg_elb < 75 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()

            # 7. CHEST FLY (Incline)
            elif exercise_name == 'inclineDumbbellChestFly':
                if le.visibility > re.visibility:
                    sh_angle = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                    el_angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                else:
                    sh_angle = calculate_angle_2d([re.x, re.y], [rs.x, rs.y], [rh.x, rh.y])
                    el_angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                
                self.debug_angles = {'Shoulder': int(sh_angle)}

                if sh_angle > 100: self.stage = "open"
                if sh_angle < 60 and self.stage == 'open':
                    self.stage = "closed"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if el_angle < 120:
                    self.form_status = "WARNING: DON'T BEND ARMS TOO MUCH"

            # 8. SQUATS (Goblet & Sumo)
            elif exercise_name in ['gobletSquat', 'sumoSquat']:
                l_knee = calculate_angle_2d([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])
                r_knee = calculate_angle_2d([rh.x, rh.y], [rk.x, rk.y], [ra.x, ra.y])
                avg_knee = (l_knee + r_knee) / 2
                
                l_torso = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                
                self.debug_angles = {'Knee': int(avg_knee)}

                if avg_knee > 160: self.stage = "up"
                if avg_knee < 100 and self.stage == 'up':
                    self.stage = "down"
                if avg_knee > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if l_torso < 50: 
                     self.form_status = "ERROR: KEEP CHEST UP"
                elif self.stage == 'down' and avg_knee > 110:
                     self.form_status = "WARNING: SQUAT DEEPER"

            # 9. HINGES (RDL & Good Morning)
            elif exercise_name in ['romanianDeadlift', 'dumbbellGoodMorning']:
                l_hip = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                r_hip = calculate_angle_2d([rs.x, rs.y], [rh.x, rh.y], [rk.x, rk.y])
                avg_hip = (l_hip + r_hip) / 2
                
                l_knee = calculate_angle_2d([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])

                self.debug_angles = {'Hip': int(avg_hip)}

                if avg_hip > 160: self.stage = "up"
                if avg_hip < 120 and self.stage == 'up':
                    self.stage = "down"
                if avg_hip > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()

                if l_knee < 130:
                     self.form_status = "ERROR: TOO MUCH KNEE BEND"
            
            # 10. BENT OVER ROW
            elif exercise_name == 'bentOverRow':
                if lw.visibility > rw.visibility:
                    el_angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                    torso_angle = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                else:
                    el_angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                    torso_angle = calculate_angle_2d([rs.x, rs.y], [rh.x, rh.y], [rk.x, rk.y])

                self.debug_angles = {'Elbow': int(el_angle)}

                if el_angle > 150: self.stage = "down"
                if el_angle < 75 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if torso_angle > 150:
                    self.form_status = "ERROR: BEND OVER MORE"

            # --- Post-Rep Processing ---
            is_new_rep = (self.rep_counter > prev_rep_counter)

            if is_new_rep:
                if "ERROR" in self.form_status:
                    self.new_error_to_log = { 
                        "rep_number": self.rep_counter, 
                        "error_type": self.form_status, 
                        "exercise_name": exercise_name 
                    }
                    if self.form_status == self.last_consecutive_error_type:
                        self.consecutive_error_counter += 1
                    else:
                        self.last_consecutive_error_type = self.form_status
                        self.consecutive_error_counter = 1
                else: 
                    self.consecutive_error_counter = 0
                    self.last_consecutive_error_type = None

                if self.consecutive_error_counter >= 6:
                    alert_message = f"Repeated Error: {self.last_consecutive_error_type.replace('ERROR: ', '')}"
                    self.triggered_alert = {'message': alert_message, 'exercise': exercise_name, 'reps': self.rep_counter}
                    self.consecutive_error_counter = 0

        except Exception as e:
            print(f"Error in analyze_frame: {e}")
            self.form_status = "ERROR: ANALYSIS FAILED"
            pass
    
    def reset_session(self):
        self.rep_counter = 0
        self.stage = None
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.triggered_alert = None
        self.new_error_to_log = None

    def get_status(self):
        return self.rep_counter, self.form_status, self.status_color, self.debug_angles
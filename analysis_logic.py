import numpy as np
import mediapipe as mp
import time
import tensorflow as tf 
from collections import deque, Counter

# --- 1. GEOMETRY HELPERS (For Angle Calculation) ---
def calculate_angle(a, b, c):
    """3D Angle: Used for complex motion analysis."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def calculate_angle_2d(a, b, c):
    """2D Angle: Used for standard form checks."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 2. FEATURE EXTRACTORS ---
def get_raw_landmarks(landmarks):
    """Returns 132 inputs: (x, y, z, vis) * 33 points."""
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row, dtype=np.float32)

def get_angle_features(landmarks):
    """Returns 8 inputs: Specific joint angles."""
    try:
        # MediaPipe indices: 11=L.Shoulder, 13=L.Elbow, 15=L.Wrist, etc.
        return np.array([
            calculate_angle(landmarks[11], landmarks[13], landmarks[15]),  # Left arm
            calculate_angle(landmarks[12], landmarks[14], landmarks[16]),  # Right arm
            calculate_angle(landmarks[13], landmarks[11], landmarks[23]),  # Left shoulder
            calculate_angle(landmarks[14], landmarks[12], landmarks[24]),  # Right shoulder
            calculate_angle(landmarks[11], landmarks[23], landmarks[25]),  # Left torso
            calculate_angle(landmarks[12], landmarks[24], landmarks[26]),  # Right torso
            calculate_angle(landmarks[23], landmarks[25], landmarks[27]),  # Left leg
            calculate_angle(landmarks[24], landmarks[26], landmarks[28])   # Right leg
        ], dtype=np.float32)
    except:
        return np.zeros(8, dtype=np.float32) # Fallback

# --- 3. ADAPTIVE ANALYZER CLASS ---
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
        
        # State for Auto-Configuration
        self.model_configured = False
        self.use_sequence = True
        self.use_angles = False # False = Raw Landmarks, True = Angles
        self.expected_seq_len = sequence_length
        
        # Buffers
        self.angle_sequence_buffer = deque(maxlen=sequence_length)
        
        # Prediction Smoothing
        self.CONF_THRESHOLD = conf_threshold
        self.STABILITY_FRAMES = stability_frames
        self.recent_predictions = deque(maxlen=self.STABILITY_FRAMES)
        self.stable_prediction = "neutral"
        
        # Error Logging
        self.triggered_alert = None
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.new_error_to_log = None
        self.frame_count = 0
        self.PREDICTION_INTERVAL = 3

    def _auto_configure_model(self, input_details):
        """
        Automatically detects if model needs Landmarks vs Angles
        and Sequence vs Single-Frame.
        """
        shape = input_details[0]['shape'] # e.g., [1, 90, 132]
        input_size = shape[-1] # Last dimension (features)
        
        print(f"--- [DEBUG] Model Configuration Detected ---")
        print(f"Input Shape: {shape}")
        
        # Detect Feature Type
        if input_size == 132:
            self.use_angles = False
            print("Feature Mode: RAW LANDMARKS (132 inputs)")
        elif input_size == 8:
            self.use_angles = True
            print("Feature Mode: ANGLES (8 inputs)")
        else:
            print(f"WARNING: Unknown input size {input_size}. Defaulting to Landmarks.")
            self.use_angles = False

        # Detect Sequence vs Single
        if len(shape) == 3 and shape[1] > 1:
            self.use_sequence = True
            self.expected_seq_len = shape[1]
            # Resize buffer to match model requirement exactly
            self.angle_sequence_buffer = deque(maxlen=self.expected_seq_len)
            print(f"Model Mode: LSTM Sequence ({self.expected_seq_len} frames)")
        else:
            self.use_sequence = False
            print("Model Mode: Single Frame (Dense)")
            
        self.model_configured = True

    def predict_with_tflite(self, interpreter, input_details, output_details, input_data):
        """Run inference handling quantization."""
        input_index = input_details[0]['index']
        input_dtype = input_details[0]['dtype']
        
        # Quantize Input if needed (Float -> Int8/Uint8)
        if input_dtype != np.float32:
            scale, zero_point = input_details[0]['quantization']
            if scale > 0:
                input_data = (input_data / scale) + zero_point
                if input_dtype == np.int8:
                    input_data = np.clip(input_data, -128, 127)
                else:
                    input_data = np.clip(input_data, 0, 255)
                input_data = input_data.astype(input_dtype)

        # Run
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        
        # De-quantize Output if needed
        output_index = output_details[0]['index']
        output_data = interpreter.get_tensor(output_index)[0]
        output_dtype = output_details[0]['dtype']
        
        if output_dtype != np.float32:
            scale, zero_point = output_details[0]['quantization']
            if scale > 0:
                output_data = (output_data.astype(np.float32) - zero_point) * scale
                
        return output_data

    def process_frame(self, interpreter, input_details, output_details, label_mapping, landmarks, current_exercise):
        self.frame_count += 1
        
        # 1. Auto-Configure on first frame
        if not self.model_configured:
            self._auto_configure_model(input_details)

        # 2. Extract Features (Adaptive)
        if self.use_angles:
            features = get_angle_features(landmarks)
        else:
            features = get_raw_landmarks(landmarks)
            
        if features is None: return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

        # 3. Manage Buffer / Input
        input_tensor = None
        
        if self.use_sequence:
            self.angle_sequence_buffer.append(features)
            # Report status while filling buffer
            if len(self.angle_sequence_buffer) < self.expected_seq_len:
                self.form_status = f"ANALYZING... {len(self.angle_sequence_buffer)}/{self.expected_seq_len}"
            elif len(self.angle_sequence_buffer) == self.expected_seq_len:
                # Shape: (1, Sequence_Len, Features)
                input_tensor = np.expand_dims(np.array(self.angle_sequence_buffer), axis=0)
        else:
            # Shape: (1, Features)
            input_tensor = np.expand_dims(features, axis=0)

        # 4. Predict (if we have valid input)
        if input_tensor is not None and self.frame_count % self.PREDICTION_INTERVAL == 0:
            try:
                prediction_output = self.predict_with_tflite(
                    interpreter, input_details, output_details, input_tensor.astype(np.float32)
                )

                # Handle Softmax if needed
                if np.max(prediction_output) > 1.0 or np.min(prediction_output) < 0.0:
                     exp_x = np.exp(prediction_output - np.max(prediction_output))
                     prediction_output = exp_x / exp_x.sum()

                predicted_idx = np.argmax(prediction_output)
                confidence = prediction_output[predicted_idx]

                if confidence > self.CONF_THRESHOLD:
                    pred_label = label_mapping.get(int(predicted_idx), "neutral")
                    self.recent_predictions.append(pred_label)
                else:
                    self.recent_predictions.append("neutral")

                # Stability Voting
                most_common, count = Counter(self.recent_predictions).most_common(1)[0]
                if count >= (self.STABILITY_FRAMES - 2):
                    self.stable_prediction = most_common

            except Exception as e:
                print(f"Prediction Error: {e}")

        # 5. Analyze Form (Logic preserved)
        self.analyze_frame(self.stable_prediction, landmarks)
        
        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if not landmarks or exercise_name == "neutral":
            if exercise_name == "neutral": 
                self.previous_exercise = "neutral"
                self.stage = None
            return

        if exercise_name != self.previous_exercise:
            self.rep_counter = 0; self.stage = None; self.previous_exercise = exercise_name
        
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

            # --- EXERCISE LOGIC BLOCKS ---

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
                # Similar to other presses but checking for full extension
                l_elb = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                r_elb = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                avg_elb = (l_elb + r_elb) / 2
                
                self.debug_angles = {'Elbow': int(avg_elb)}

                if avg_elb < 90: self.stage = "in"
                if avg_elb > 160 and self.stage == 'in':
                    self.stage = "out"; self.rep_counter += 1; self.last_rep_time = time.time()

            # 6. UPRIGHT ROW
            elif exercise_name == 'uprightRow':
                # Elbows flare out and up. Angle at elbow closes.
                l_elb = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                r_elb = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                avg_elb = (l_elb + r_elb) / 2
                
                self.debug_angles = {'Elbow': int(avg_elb)}

                if avg_elb > 140: self.stage = "down"
                if avg_elb < 75 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()

            # 7. CHEST FLY (Incline)
            elif exercise_name == 'inclineDumbbellChestFly':
                # Shoulder angle changes (arms open), Elbow angle stays constant
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
                
                # Torso lean check
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

    def get_triggered_alert(self):
        alert = self.triggered_alert; self.triggered_alert = None; return alert

    def get_new_error_log(self):
        log = self.new_error_to_log; self.new_error_to_log = None; return log

    def reset_session(self):
        self.rep_counter = 0; self.stage = None; self.angle_sequence_buffer.clear()
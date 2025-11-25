import numpy as np
import time
import math
from collections import deque, Counter

# --- 1. GEOMETRY & NORMALIZATION ---

def normalize_pose_robust(landmarks):
    """
    Normalizes landmarks based on torso length. 
    Returns normalized (33, 3) array or None.
    """
    try:
        lms = []
        for lm in landmarks:
            if isinstance(lm, dict):
                lms.append([float(lm['x']), float(lm['y']), float(lm.get('z', 0.0))])
            else:
                lms.append([float(lm.x), float(lm.y), float(lm.z)])
        
        landmarks_np = np.array(lms, dtype=np.float32)

        # Indices: Left Hip=23, Right Hip=24, Left Shoulder=11, Right Shoulder=12
        left_hip = landmarks_np[23]
        right_hip = landmarks_np[24]
        hip_center = (left_hip + right_hip) / 2.0

        left_shoulder = landmarks_np[11]
        right_shoulder = landmarks_np[12]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        torso_length = np.linalg.norm(hip_center - shoulder_center) + 1e-6
        
        if torso_length < 1e-5: return None

        normalized_landmarks = (landmarks_np - hip_center) / torso_length
        return normalized_landmarks
    except Exception as e:
        print(f"Norm Error: {e}")
        return None

def calculate_angle_3d(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0: return 0.0
    
    dot_product = np.dot(ba, bc)
    cosine_angle = dot_product / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_angle_2d(a, b, c):
    a = np.array(a[:2]); b = np.array(b[:2]); c = np.array(c[:2])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 2. FEATURE ENGINEERING (Matches Model Input) ---

def extract_engineered_features(landmarks):
    """Extracts 42 engineered features + 5 pad = 47 total."""
    norm_lms = normalize_pose_robust(landmarks)
    if norm_lms is None: return None

    def lm(i): return norm_lms[i]

    # 1. Angles (20)
    angles = [
        calculate_angle_3d(lm(11), lm(23), lm(25)), calculate_angle_3d(lm(12), lm(24), lm(26)),
        calculate_angle_3d(lm(11), lm(24), lm(12)), calculate_angle_3d(lm(0), lm(7), lm(8)),
        calculate_angle_3d(lm(23), lm(11), lm(12)), calculate_angle_3d(lm(24), lm(12), lm(11)),
        calculate_angle_3d(lm(11), lm(13), lm(15)), calculate_angle_3d(lm(12), lm(14), lm(16)),
        calculate_angle_3d(lm(23), lm(11), lm(13)), calculate_angle_3d(lm(24), lm(12), lm(14)),
        calculate_angle_3d(lm(13), lm(15), lm(19)), calculate_angle_3d(lm(14), lm(16), lm(20)),
        calculate_angle_3d(lm(11), lm(23), lm(25)), calculate_angle_3d(lm(12), lm(24), lm(26)),
        calculate_angle_3d(lm(23), lm(25), lm(27)), calculate_angle_3d(lm(24), lm(26), lm(28)),
        calculate_angle_3d(lm(25), lm(27), lm(29)), calculate_angle_3d(lm(26), lm(28), lm(30)),
        calculate_angle_3d(lm(12), lm(23), lm(24)), calculate_angle_3d(lm(11), lm(24), lm(23))
    ]

    # 2. Distances (22)
    def dist(i, j): return np.linalg.norm(lm(i) - lm(j))
    hip_center = np.array([0.0, 0.0, 0.0]) 

    distances = [
        dist(11, 12), dist(23, 24), dist(15, 25), dist(16, 26),
        dist(13, 23), dist(14, 24), dist(27, 15), dist(28, 16),
        np.linalg.norm(lm(0) - hip_center),
        abs(lm(15)[1] - lm(11)[1]), abs(lm(16)[1] - lm(12)[1]),
        abs(lm(23)[1] - lm(25)[1]), abs(lm(24)[1] - lm(26)[1]),
        abs(lm(11)[1] - lm(23)[1]), abs(lm(12)[1] - lm(24)[1]),
        abs(lm(27)[1] - lm(29)[1]), abs(lm(28)[1] - lm(30)[1]),
        abs(lm(15)[2] - lm(23)[2]), abs(lm(16)[2] - lm(24)[2]),
        abs(lm(11)[2] - lm(23)[2]), abs(lm(12)[2] - lm(24)[2]),
        abs(lm(0)[2] - hip_center[2])
    ]

    features = np.array(angles + distances, dtype=np.float32)
    
    # 3. Padding to 47 (Robust Method)
    if len(features) == 42:
        features = np.concatenate([features, np.zeros(5, dtype=np.float32)])
        
    return features

# --- 3. ANALYZER CLASS ---

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
        
        self.model_configured = False
        self.expected_seq_len = sequence_length
        self.input_size = 0
        self.angle_sequence_buffer = deque(maxlen=sequence_length)
        self.CONF_THRESHOLD = conf_threshold
        self.STABILITY_FRAMES = stability_frames
        self.recent_predictions = deque(maxlen=self.STABILITY_FRAMES)
        self.stable_prediction = "neutral"
        self.frame_count = 0
        self.PREDICTION_INTERVAL = 3
        
        self.triggered_alert = None
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.new_error_to_log = None

    def _auto_configure_model(self, input_details):
        shape = input_details[0]['shape']
        self.input_size = int(shape[-1])
        
        print(f"\n--- [DEBUG] MODEL CONFIG ---")
        print(f"Input Shape: {shape}")
        
        if len(shape) == 3:
            self.expected_seq_len = int(shape[1])
            self.angle_sequence_buffer = deque(maxlen=self.expected_seq_len)
        
        self.model_configured = True

    def predict_with_tflite(self, interpreter, input_details, output_details, input_data):
        input_index = input_details[0]['index']
        input_dtype = input_details[0]['dtype']
        
        if input_dtype != np.float32:
            scale, zero_point = input_details[0]['quantization']
            if scale > 0:
                input_data = (input_data / scale) + zero_point
                if input_dtype == np.int8:
                    input_data = np.clip(input_data, -128, 127)
                else:
                    input_data = np.clip(input_data, 0, 255)
                input_data = input_data.astype(input_dtype)

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        
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
        
        if not self.model_configured:
            self._auto_configure_model(input_details)

        # Extract Features
        if self.input_size == 47:
            features = extract_engineered_features(landmarks)
        elif self.input_size == 132:
            # Fallback for raw landmarks
            lms = []
            for lm in landmarks:
                v = lm['visibility'] if isinstance(lm, dict) else lm.visibility
                x = lm['x'] if isinstance(lm, dict) else lm.x
                y = lm['y'] if isinstance(lm, dict) else lm.y
                z = lm.get('z', 0.0) if isinstance(lm, dict) else lm.z
                lms.extend([x, y, z, v])
            features = np.array(lms, dtype=np.float32)
        else:
            features = np.zeros(self.input_size, dtype=np.float32)

        if features is None: 
            return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

        self.angle_sequence_buffer.append(features)
        
        # Predict
        if len(self.angle_sequence_buffer) == self.expected_seq_len and self.frame_count % self.PREDICTION_INTERVAL == 0:
            try:
                input_tensor = np.expand_dims(np.array(self.angle_sequence_buffer), axis=0)
                
                prediction_output = self.predict_with_tflite(
                    interpreter, input_details, output_details, input_tensor.astype(np.float32)
                )

                if np.max(prediction_output) > 1.0: # Softmax if needed
                     exp_x = np.exp(prediction_output - np.max(prediction_output))
                     prediction_output = exp_x / exp_x.sum()

                predicted_idx = np.argmax(prediction_output)
                confidence = prediction_output[predicted_idx]
                
                # Safe Label Lookup
                safe_idx = int(predicted_idx)
                pred_label = label_mapping.get(safe_idx, label_mapping.get(str(safe_idx), "neutral"))

                # Debug Print (Uncomment if needed)
                # print(f"Pred: {pred_label} ({confidence:.2f})")

                if confidence > self.CONF_THRESHOLD:
                    self.recent_predictions.append(pred_label)
                else:
                    self.recent_predictions.append("neutral")

                most_common, count = Counter(self.recent_predictions).most_common(1)[0]
                if count >= (self.STABILITY_FRAMES - 2):
                    self.stable_prediction = most_common

            except Exception as e:
                print(f"Inference Error: {e}")

        self.analyze_frame(self.stable_prediction, landmarks)
        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if not landmarks: return
        
        # Convert to objects for simple logic
        class Point:
            def __init__(self, obj):
                self.x = float(obj['x']) if isinstance(obj, dict) else float(obj.x)
                self.y = float(obj['y']) if isinstance(obj, dict) else float(obj.y)
        
        lms_obj = [Point(lm) for lm in landmarks]

        if exercise_name == "neutral":
            if self.previous_exercise != "neutral":
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
            
            # --- LANDMARKS ---
            ls = lms_obj[11]; le = lms_obj[13]; lw = lms_obj[15] # Left Arm
            rs = lms_obj[12]; re = lms_obj[14]; rw = lms_obj[16] # Right Arm
            lh = lms_obj[23]; lk = lms_obj[25]; la = lms_obj[27] # Left Leg
            rh = lms_obj[24]; rk = lms_obj[26]; ra = lms_obj[28] # Right Leg

            # --- EXERCISE LOGIC ---

            # 1. BICEP CURL
            if exercise_name == 'bicepCurl':
                if lw.x < ls.x: # Left side visible
                    angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                    sh_angle = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                else:
                    angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                    sh_angle = calculate_angle_2d([re.x, re.y], [rs.x, rs.y], [rh.x, rh.y])
                
                self.debug_angles = {'Elbow': int(angle)}
                
                if angle > 160: self.stage = "down"
                if angle < 45 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if sh_angle > 45:
                    self.form_status = "ERROR: ELBOWS SWINGING"; self.status_color = (0, 0, 255)

            # 2. TRICEP KICKBACK
            elif exercise_name == 'tricepKickback':
                if lw.x < ls.x:
                    angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                    torso = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                else:
                    angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                    torso = calculate_angle_2d([rs.x, rs.y], [rh.x, rh.y], [rk.x, rk.y])
                
                self.debug_angles = {'Elbow': int(angle)}
                
                if angle < 90: self.stage = "in"
                if angle > 160 and self.stage == 'in':
                    self.stage = "out"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if torso > 135: self.form_status = "ERROR: BEND OVER MORE"

            # 3. LATERAL RAISE & REVERSE FLY
            elif exercise_name in ['lateralRaise', 'dumbbellReverseFly']:
                if le.x < ls.x:
                    angle = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                else:
                    angle = calculate_angle_2d([re.x, re.y], [rs.x, rs.y], [rh.x, rh.y])
                
                self.debug_angles = {'Shoulder': int(angle)}

                if angle < 30: self.stage = "down"
                if angle > 80 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                if angle > 100: self.form_status = "WARNING: DO NOT OVER-RAISE"

            # 4. PRESSES (Shoulder / Incline)
            elif exercise_name in ['shoulderPress', 'dumbbellPushPress', 'inclineBenchPress']:
                l_elb = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                r_elb = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                avg_elb = (l_elb + r_elb) / 2
                
                self.debug_angles = {'Elbow': int(avg_elb)}

                if avg_elb < 80: self.stage = "down"
                if avg_elb > 150 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()

            # 5. SVEND PRESS
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

            # 7. CHEST FLY
            elif exercise_name == 'inclineDumbbellChestFly':
                if le.x < ls.x:
                    sh_angle = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                    el_angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                else:
                    sh_angle = calculate_angle_2d([re.x, re.y], [rs.x, rs.y], [rh.x, rh.y])
                    el_angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                
                self.debug_angles = {'Shoulder': int(sh_angle)}

                if sh_angle > 100: self.stage = "open"
                if sh_angle < 60 and self.stage == 'open':
                    self.stage = "closed"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if el_angle < 120: self.form_status = "WARNING: DON'T BEND ARMS TOO MUCH"

            # 8. SQUATS
            elif exercise_name in ['gobletSquat', 'sumoSquat']:
                l_knee = calculate_angle_2d([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])
                r_knee = calculate_angle_2d([rh.x, rh.y], [rk.x, rk.y], [ra.x, ra.y])
                avg_knee = (l_knee + r_knee) / 2
                l_torso = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                
                self.debug_angles = {'Knee': int(avg_knee)}

                if avg_knee > 160: self.stage = "up"
                if avg_knee < 100 and self.stage == 'up': self.stage = "down"
                if avg_knee > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if l_torso < 50: self.form_status = "ERROR: KEEP CHEST UP"
                elif self.stage == 'down' and avg_knee > 110: self.form_status = "WARNING: SQUAT DEEPER"

            # 9. HINGES (RDL / Good Morning)
            elif exercise_name in ['romanianDeadlift', 'dumbbellGoodMorning']:
                l_hip = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                r_hip = calculate_angle_2d([rs.x, rs.y], [rh.x, rh.y], [rk.x, rk.y])
                avg_hip = (l_hip + r_hip) / 2
                l_knee = calculate_angle_2d([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])

                self.debug_angles = {'Hip': int(avg_hip)}

                if avg_hip > 160: self.stage = "up"
                if avg_hip < 120 and self.stage == 'up': self.stage = "down"
                if avg_hip > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()

                if l_knee < 130: self.form_status = "ERROR: TOO MUCH KNEE BEND"

            # 10. BENT OVER ROW
            elif exercise_name == 'bentOverRow':
                if lw.x < ls.x:
                    el_angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                    torso = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                else:
                    el_angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                    torso = calculate_angle_2d([rs.x, rs.y], [rh.x, rh.y], [rk.x, rk.y])

                self.debug_angles = {'Elbow': int(el_angle)}

                if el_angle > 150: self.stage = "down"
                if el_angle < 75 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1; self.last_rep_time = time.time()
                
                if torso > 150: self.form_status = "ERROR: BEND OVER MORE"

            # --- LOGGING ---
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
                    msg = f"Repeated Error: {self.last_consecutive_error_type.replace('ERROR: ', '')}"
                    self.triggered_alert = {'message': msg, 'exercise': exercise_name, 'reps': self.rep_counter}
                    self.consecutive_error_counter = 0

        except Exception:
            pass

    def get_triggered_alert(self):
        alert = self.triggered_alert; self.triggered_alert = None; return alert

    def get_new_error_log(self):
        log = self.new_error_to_log; self.new_error_to_log = None; return log

    def reset_session(self):
        self.rep_counter = 0; self.stage = None; self.angle_sequence_buffer.clear()
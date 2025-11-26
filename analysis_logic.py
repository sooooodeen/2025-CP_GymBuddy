import numpy as np
import time
import math
from collections import deque, Counter

# --- 1. GEOMETRY & NORMALIZATION ---

def normalize_pose_robust(landmarks):
    """
    Normalizes landmarks based on torso length. 
    Matches feature_engineering.py logic.
    """
    try:
        lms = []
        for lm in landmarks:
            if isinstance(lm, dict):
                lms.append([float(lm['x']), float(lm['y']), float(lm.get('z', 0.0))])
            else:
                lms.append([float(lm.x), float(lm.y), float(lm.z)])
        
        landmarks_np = np.array(lms, dtype=np.float32)

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
    """Calculates 3D angle."""
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
    """2D Angle helper for Form Checks."""
    a = np.array(a[:2]); b = np.array(b[:2]); c = np.array(c[:2])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 2. FEATURE ENGINEERING ---

def extract_engineered_features(landmarks):
    """Extracts exactly 47 features (42 calculated + 5 padding)."""
    norm_lms = normalize_pose_robust(landmarks)
    if norm_lms is None: return None

    def lm(i): return norm_lms[i]

    # --- Angles (20) ---
    angles = [
        calculate_angle_3d(lm(11), lm(23), lm(25)), # L Torso Side
        calculate_angle_3d(lm(12), lm(24), lm(26)), # R Torso Side
        calculate_angle_3d(lm(11), lm(24), lm(12)), # Torso Front
        calculate_angle_3d(lm(0), lm(7), lm(8)),    # Neck
        calculate_angle_3d(lm(23), lm(11), lm(12)), # Spine L
        calculate_angle_3d(lm(24), lm(12), lm(11)), # Spine R
        calculate_angle_3d(lm(11), lm(13), lm(15)), # L Elbow
        calculate_angle_3d(lm(12), lm(14), lm(16)), # R Elbow
        calculate_angle_3d(lm(23), lm(11), lm(13)), # L Shoulder Abd
        calculate_angle_3d(lm(24), lm(12), lm(14)), # R Shoulder Abd
        calculate_angle_3d(lm(13), lm(15), lm(19)), # L Wrist
        calculate_angle_3d(lm(14), lm(16), lm(20)), # R Wrist
        calculate_angle_3d(lm(11), lm(23), lm(25)), # L Hip
        calculate_angle_3d(lm(12), lm(24), lm(26)), # R Hip
        calculate_angle_3d(lm(23), lm(25), lm(27)), # L Knee
        calculate_angle_3d(lm(24), lm(26), lm(28)), # R Knee
        calculate_angle_3d(lm(25), lm(27), lm(29)), # L Ankle
        calculate_angle_3d(lm(26), lm(28), lm(30)), # R Ankle
        calculate_angle_3d(lm(12), lm(23), lm(24)), # Twist L
        calculate_angle_3d(lm(11), lm(24), lm(23))  # Twist R
    ]

    # --- Distances (22) ---
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
        self.expected_seq_len = int(sequence_length)
        self.input_size = 0
        self.angle_sequence_buffer = deque(maxlen=self.expected_seq_len)
        self.CONF_THRESHOLD = conf_threshold
        self.STABILITY_FRAMES = int(stability_frames)
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
        print(f"\n--- [DEBUG] MODEL CONFIG: Input Shape: {shape} ---")
        
        if len(shape) == 3:
            self.expected_seq_len = int(shape[1])
            self.angle_sequence_buffer = deque(maxlen=self.expected_seq_len)
        self.model_configured = True

    def predict_with_tflite(self, interpreter, input_details, output_details, input_data):
        input_index = input_details[0]['index']
        if input_details[0]['dtype'] != np.float32:
            scale, zero_point = input_details[0]['quantization']
            if scale > 0:
                input_data = (input_data / scale) + zero_point
                input_data = np.clip(input_data, -128, 127) if input_details[0]['dtype'] == np.int8 else np.clip(input_data, 0, 255)
                input_data = input_data.astype(input_details[0]['dtype'])

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]

    def _apply_logic_override(self, ai_prediction, landmarks):
        """Fixes common model mistakes using 2D geometry."""
        if not landmarks: return ai_prediction
        def get_x(i): return landmarks[i]['x'] if isinstance(landmarks[i], dict) else landmarks[i].x
        
        # Fix: If predicted Fly/Lateral Raise, but hands are close to shoulders (horizontally) -> It's a Curl/Press
        if ai_prediction in ['dumbbellReverseFly', 'lateralRaise', 'inclineDumbbellChestFly']:
            l_width = abs(get_x(15) - get_x(11))
            r_width = abs(get_x(16) - get_x(12))
            if l_width < 0.20 and r_width < 0.20: return 'bicepCurl'

        return ai_prediction

    def process_frame(self, interpreter, input_details, output_details, label_mapping, landmarks, current_exercise):
        self.frame_count += 1
        
        if not self.model_configured: self._auto_configure_model(input_details)

        if self.input_size == 47: features = extract_engineered_features(landmarks)
        else: features = np.zeros(self.input_size, dtype=np.float32)

        if features is None: 
            return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

        self.angle_sequence_buffer.append(features)
        
        if len(self.angle_sequence_buffer) == self.expected_seq_len and self.frame_count % self.PREDICTION_INTERVAL == 0:
            try:
                input_tensor = np.expand_dims(np.array(self.angle_sequence_buffer), axis=0)
                prediction = self.predict_with_tflite(interpreter, input_details, output_details, input_tensor.astype(np.float32))
                
                if np.max(prediction) > 1.0:
                     prediction = np.exp(prediction - np.max(prediction))
                     prediction = prediction / prediction.sum()

                idx = int(np.argmax(prediction))
                conf = prediction[idx]
                
                # Safe Label Lookup
                pred_label = str(label_mapping.get(idx, label_mapping.get(str(idx), "neutral")))

                # Logic Patch
                final_label = self._apply_logic_override(pred_label, landmarks)
                
                if conf > self.CONF_THRESHOLD: self.recent_predictions.append(final_label)
                else: self.recent_predictions.append("neutral")

                most_common, count = Counter(self.recent_predictions).most_common(1)[0]
                if count >= (self.STABILITY_FRAMES - 2): self.stable_prediction = most_common

            except Exception as e: print(f"Inference Error: {e}")

        self.analyze_frame(self.stable_prediction, landmarks)
        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if not landmarks: return
        
        class Point:
            def __init__(self, obj):
                self.x = float(obj['x']) if isinstance(obj, dict) else float(obj.x)
                self.y = float(obj['y']) if isinstance(obj, dict) else float(obj.y)
        
        lms = [Point(lm) for lm in landmarks]

        if exercise_name == "neutral":
            if self.previous_exercise != "neutral":
                self.previous_exercise = "neutral"; self.stage = None
            return

        if exercise_name != self.previous_exercise:
            self.rep_counter = 0; self.stage = None; self.previous_exercise = exercise_name
        
        self.last_rep_time = time.time()
        self.form_status = "CORRECT FORM"
        self.status_color = (0, 255, 0)

        try:
            # --- Landmark Aliases ---
            ls, rs = lms[11], lms[12] # Shoulders
            le, re = lms[13], lms[14] # Elbows
            lw, rw = lms[15], lms[16] # Wrists
            lh, rh = lms[23], lms[24] # Hips
            lk, rk = lms[25], lms[26] # Knees
            la, ra = lms[27], lms[28] # Ankles

            # --- LOGIC FOR ALL 16 EXERCISES ---

            # 0. Bent Over Row
            if exercise_name == 'bentOverRow':
                # Check Elbow Angle (Pulling vs Extended)
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                # Check Back Angle (Shoulder-Hip-Knee) - Should be bent (~45-90 deg from vertical)
                hip_ang = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                
                self.debug_angles = {'Elbow': int(elb_ang), 'Hip': int(hip_ang)}
                
                if elb_ang > 150: self.stage = "down"
                if elb_ang < 80 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1
                
                if hip_ang > 150: self.form_status = "ERROR: BEND OVER MORE"

            # 1. Bicep Curl
            elif exercise_name == 'bicepCurl':
                angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                sh_angle = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                self.debug_angles = {'Elbow': int(angle)}
                if angle > 160: self.stage = "down"
                if angle < 45 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1
                if sh_angle > 40: self.form_status = "ERROR: ELBOWS SWINGING"

            # 2. Dumbbell Good Morning
            elif exercise_name == 'dumbbellGoodMorning':
                hip_ang = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                self.debug_angles = {'Hip': int(hip_ang)}
                if hip_ang > 160: self.stage = "up"
                if hip_ang < 110 and self.stage == 'up': self.stage = "down"
                if hip_ang > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1

            # 3. Dumbbell Push Press (Uses legs)
            elif exercise_name == 'dumbbellPushPress':
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                self.debug_angles = {'Elbow': int(elb_ang)}
                if elb_ang < 70: self.stage = "down"
                if elb_ang > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1

            # 4. Dumbbell Reverse Fly
            elif exercise_name == 'dumbbellReverseFly':
                # Check Angle between Arms (horizontal abduction)
                # Simplified: check wrist distance
                dist = abs(lw.x - rw.x)
                if dist < 0.2: self.stage = "in"
                if dist > 0.6 and self.stage == 'in':
                    self.stage = "out"; self.rep_counter += 1
                
                # Form Check: Elbows shouldn't be too bent (like a row)
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                if elb_ang < 100: self.form_status = "ERROR: ARMS TOO BENT"

            # 5. Dumbbell Svend Press
            elif exercise_name == 'dumbbellSvendPress':
                # Pressing outwards from chest
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                if elb_ang < 70: self.stage = "in"
                if elb_ang > 160 and self.stage == 'in':
                    self.stage = "out"; self.rep_counter += 1

            # 6. Goblet Squat / 13. Sumo Squat
            elif exercise_name in ['gobletSquat', 'sumoSquat']:
                knee_ang = calculate_angle_2d([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])
                torso_ang = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                self.debug_angles = {'Knee': int(knee_ang)}
                if knee_ang > 160: self.stage = "up"
                if knee_ang < 100 and self.stage == 'up': self.stage = "down"
                if knee_ang > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1
                if torso_ang < 60: self.form_status = "ERROR: CHEST UP"

            # 7. Incline Bench Press
            elif exercise_name == 'inclineBenchPress':
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                self.debug_angles = {'Elbow': int(elb_ang)}
                if elb_ang < 70: self.stage = "down"
                if elb_ang > 160 and self.stage == "down":
                    self.stage = "up"; self.rep_counter += 1

            # 8. Incline Dumbbell Chest Fly
            elif exercise_name == 'inclineDumbbellChestFly':
                # Shoulder Angle (Abduction)
                sh_ang = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                if sh_ang < 45: self.stage = "up" # Hands together above chest
                if sh_ang > 75 and self.stage == 'up': self.stage = "down" # Hands wide
                if sh_ang < 45 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1
                
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                if elb_ang < 100: self.form_status = "ERROR: ARMS TOO BENT"

            # 9. Lateral Raise
            elif exercise_name == 'lateralRaise':
                sh_ang = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                self.debug_angles = {'Shoulder': int(sh_ang)}
                if sh_ang < 30: self.stage = "down"
                if sh_ang > 80 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1
                if sh_ang > 100: self.form_status = "WARNING: TOO HIGH"

            # 11. Romanian Deadlift
            elif exercise_name == 'romanianDeadlift':
                hip_ang = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                knee_ang = calculate_angle_2d([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])
                self.debug_angles = {'Hip': int(hip_ang)}
                if hip_ang > 160: self.stage = "up"
                if hip_ang < 120 and self.stage == 'up': self.stage = "down"
                if hip_ang > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1
                if knee_ang < 140: self.form_status = "ERROR: TOO MUCH KNEE BEND"

            # 12. Shoulder Press
            elif exercise_name == 'shoulderPress':
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                if elb_ang < 70: self.stage = "down"
                if elb_ang > 150 and self.stage == "down":
                    self.stage = "up"; self.rep_counter += 1

            # 14. Tricep Kickback
            elif exercise_name == 'tricepKickback':
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                # Torso should be bent
                torso_ang = calculate_angle_2d([ls.x, ls.y], [lh.x, lh.y], [lk.x, lk.y])
                
                self.debug_angles = {'Elbow': int(elb_ang)}
                if elb_ang < 90: self.stage = "in"
                if elb_ang > 160 and self.stage == "in":
                    self.stage = "out"; self.rep_counter += 1
                if torso_ang > 135: self.form_status = "ERROR: BEND OVER"

            # 15. Upright Row
            elif exercise_name == 'uprightRow':
                # Elbows flex (angle decreases) as weight comes up
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                self.debug_angles = {'Elbow': int(elb_ang)}
                if elb_ang > 150: self.stage = "down"
                if elb_ang < 75 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1
                
                # Check if wrists are higher than elbows (Bad form)
                if lw.y < le.y: self.form_status = "ERROR: ELBOWS HIGHER"

            # --- LOGGING ---
            if (self.rep_counter > prev_rep_counter):
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

        except Exception as e:
            print(f"Logic Error: {e}")
            pass

    def get_triggered_alert(self):
        alert = self.triggered_alert; self.triggered_alert = None; return alert

    def get_new_error_log(self):
        log = self.new_error_to_log; self.new_error_to_log = None; return log

    def reset_session(self):
        self.rep_counter = 0; self.stage = None; self.angle_sequence_buffer.clear()
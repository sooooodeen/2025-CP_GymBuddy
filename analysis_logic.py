import numpy as np
import time
import math
from collections import deque, Counter

# --- 1. GEOMETRY ENGINE (3D FIXES) ---

def calculate_angle_3d(a, b, c):
    """Calculates the 3D angle at point b using vector dot products."""
    a = np.array([a.x, a.y, a.z]); b = np.array([b.x, b.y, b.z]); c = np.array([c.x, c.y, c.z])
    ba = a - b; bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_inclination_3d(point_top, point_bottom):
    """Calculates verticality relative to gravity (0=Vertical Down, 90=Horizontal)."""
    p1 = np.array([point_top.x, point_top.y, point_top.z])
    p2 = np.array([point_bottom.x, point_bottom.y, point_bottom.z])
    vector = p2 - p1
    unit_vector = vector / (np.linalg.norm(vector) + 1e-7)
    dot_product = np.dot(unit_vector, [0, 1, 0]) # Compare to vertical Y axis
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    return abs(angle - 180)

# --- 2. DATA PREPROCESSING (Maintains Model Compatibility) ---

def normalize_pose_robust(landmarks):
    try:
        lms = []
        for lm in landmarks:
            if isinstance(lm, dict): lms.append([float(lm['x']), float(lm['y']), float(lm.get('z', 0.0))])
            else: lms.append([float(lm.x), float(lm.y), float(lm.z)])
        landmarks_np = np.array(lms, dtype=np.float32)
        hip_center_3d = (landmarks_np[23] + landmarks_np[24]) / 2.0
        torso_len = np.linalg.norm(landmarks_np[11] - landmarks_np[23]) + 1e-6
        return (landmarks_np - hip_center_3d) / torso_len
    except: return None

def extract_engineered_features(landmarks):
    norm_lms = normalize_pose_robust(landmarks)
    if norm_lms is None: return None
    def lm(i): return norm_lms[i]
    def ang_2d(a,b,c):
        v1 = lm(a)[:2] - lm(b)[:2]; v2 = lm(c)[:2] - lm(b)[:2]
        res = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
        return abs(res) if abs(res) <= 180 else 360 - abs(res)
    def dist(i, j): return np.linalg.norm(lm(i) - lm(j))
    
    angles = [
        ang_2d(11,23,25), ang_2d(12,24,26), ang_2d(11,24,12), ang_2d(0,7,8), ang_2d(23,11,12), ang_2d(24,12,11),
        ang_2d(11,13,15), ang_2d(12,14,16), ang_2d(23,11,13), ang_2d(24,12,14), ang_2d(13,15,19), ang_2d(14,16,20),
        ang_2d(11,23,25), ang_2d(12,24,26), ang_2d(23,25,27), ang_2d(24,26,28), ang_2d(25,27,29), ang_2d(26,28,30),
        ang_2d(12,23,24), ang_2d(11,24,23)
    ]
    hip_center = np.array([0.0, 0.0, 0.0])
    distances = [
        dist(11, 12), dist(23, 24), dist(15, 25), dist(16, 26), dist(13, 23), dist(14, 24), dist(27, 15), dist(28, 16),
        np.linalg.norm(lm(0) - hip_center),
        abs(lm(15)[1] - lm(11)[1]), abs(lm(16)[1] - lm(12)[1]), abs(lm(23)[1] - lm(25)[1]), abs(lm(24)[1] - lm(26)[1]),
        abs(lm(11)[1] - lm(23)[1]), abs(lm(12)[1] - lm(24)[1]), abs(lm(27)[1] - lm(29)[1]), abs(lm(28)[1] - lm(30)[1]),
        abs(lm(15)[2] - lm(23)[2]), abs(lm(16)[2] - lm(24)[2]), abs(lm(11)[2] - lm(23)[2]), abs(lm(12)[2] - lm(24)[2]),
        abs(lm(0)[2] - hip_center[2])
    ]
    features = np.array(angles + distances, dtype=np.float32)
    if len(features) < 47: features = np.concatenate([features, np.zeros(47 - len(features))])
    return features

# --- 3. EXERCISE ANALYZER ---

class ExerciseAnalyzer:
    def __init__(self, sequence_length=45, conf_threshold=0.60, stability_frames=10, reset_timeout=5.0):
        # Rep Counting & Status
        self.rep_counter = 0; self.stage = None; self.form_status = "START EXERCISE"; self.status_color = (0, 255, 0)
        self.last_rep_time = time.time(); self.RESET_TIMEOUT = reset_timeout
        
        # AI Model Configuration
        self.model_configured = False; self.expected_seq_len = int(sequence_length); self.input_size = 0
        self.angle_sequence_buffer = deque(maxlen=self.expected_seq_len)
        self.CONF_THRESHOLD = conf_threshold
        
        # Stability & Locking (INCREASED STABILITY)
        self.STABILITY_FRAMES = stability_frames
        self.recent_predictions = deque(maxlen=self.STABILITY_FRAMES); self.stable_prediction = "neutral"
        self.locked_exercise = None; self.neutral_persistence_counter = 0
        self.frame_count = 0; self.PREDICTION_INTERVAL = 2; self.stable_counter = 0
        
        # Error Logging
        self.triggered_alert = None; self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None; self.new_error_to_log = None
        self.debug_angles = {}

    class Point:
        def __init__(self, lm):
            self.x = lm['x'] if isinstance(lm, dict) else lm.x
            self.y = lm['y'] if isinstance(lm, dict) else lm.y
            self.z = lm['z'] if isinstance(lm, dict) else lm.z

    def _auto_configure_model(self, input_details):
        shape = input_details[0]['shape']; self.input_size = int(shape[-1])
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
        interpreter.set_tensor(input_index, input_data); interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]

    def _apply_logic_override(self, ai_prediction, landmarks):
        """Forces Neutral if the pose makes the detected exercise physically impossible."""
        if not landmarks: return ai_prediction
        p = [self.Point(lm) for lm in landmarks]
        ls, rs, lw, rw, nose = p[11], p[12], p[15], p[16], p[0]
        lh, rh = p[23], p[24]
        
        # --- 1. THE "HEIGHT" GUARD (Stops Upright Rows / Presses when hands are low) ---
        # If hands are below the navel (Hip Y), you cannot be doing a Press or Upright Row.
        hands_low = (lw.y > lh.y) and (rw.y > rh.y)
        
        if ai_prediction in ["uprightRow", "shoulderPress", "lateralRaise"]:
            if hands_low: return "neutral"

        # --- 2. THE "KICKBACK" GUARD (The Ghost Detection Killer) ---
        # Kickbacks & Rows require the Torso to be bent over.
        torso_inc = calculate_inclination_3d(ls, lh)
        
        # Calculate Upper Arm Inclination (Is the arm raised behind/horizontal?)
        # We construct a point for the Elbows
        le_pt, re_pt = p[13], p[14]
        l_uarm_inc = calculate_inclination_3d(ls, le_pt) 
        r_uarm_inc = calculate_inclination_3d(rs, re_pt)
        max_uarm_inc = max(l_uarm_inc, r_uarm_inc)

        if ai_prediction in ["tricepKickback", "bentOverRow", "dumbbellReverseFly"]:
            # If standing up straight (angle < 45), it's NOT a kickback/row
            if torso_inc < 45: return "neutral"
            
            # Additional Kickback check: Upper arm must be somewhat horizontal
            if ai_prediction == "tricepKickback" and max_uarm_inc < 50:
                return "neutral"

        # --- 3. BICEP CURL GUARD ---
        # If hands are very wide (like a T-pose), it's a Lateral Raise, not a Curl.
        is_wide = abs(lw.x - rw.x) > (abs(ls.x - rs.x) * 1.6)
        if ai_prediction == "bicepCurl" and is_wide:
            return "lateralRaise"

        return ai_prediction

    def process_frame(self, interpreter, input_details, output_details, label_mapping, landmarks, current_exercise, scaler=None):
        self.frame_count += 1
        if not self.model_configured: self._auto_configure_model(input_details)
        
        features = extract_engineered_features(landmarks) if self.input_size == 47 else np.zeros(self.input_size, dtype=np.float32)
        if features is None: return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles
        
        if scaler:
            try: features = scaler.transform(features.reshape(1, -1)).flatten()
            except: pass
        self.angle_sequence_buffer.append(features)

        # PREDICT only when buffer is full
        if len(self.angle_sequence_buffer) == self.expected_seq_len and self.frame_count % self.PREDICTION_INTERVAL == 0:
            try:
                input_tensor = np.expand_dims(np.array(self.angle_sequence_buffer), axis=0).astype(np.float32)
                prediction = self.predict_with_tflite(interpreter, input_details, output_details, input_tensor)
                
                idx = int(np.argmax(prediction))
                conf = prediction[idx]

                # --- CONFIDENCE FILTER ---
                # If the AI is unsure (< 70%), assume Neutral.
                if conf < 0.70: 
                    raw_label = "neutral"
                else:
                    raw_label = str(label_mapping.get(idx, "neutral"))
                
                final_label = self._apply_logic_override(raw_label, landmarks)
                
                # Locking Logic
                if self.locked_exercise:
                    if final_label == "neutral":
                        self.neutral_persistence_counter += 1
                        if self.neutral_persistence_counter > 40: # Faster unlock
                            self.locked_exercise = None; self.rep_counter = 0; final_label = "neutral"
                        else: final_label = self.locked_exercise
                    else: self.neutral_persistence_counter = 0; final_label = self.locked_exercise

                self.recent_predictions.append(final_label)
                most_common, count = Counter(self.recent_predictions).most_common(1)[0]
                
                # Increased stability requirement (Count > 6 out of 10)
                if count > 6:
                    if self.stable_prediction != most_common:
                        if not self.locked_exercise: self.rep_counter = 0; self.stage = None
                        self.stable_prediction = most_common; self.stable_counter = 0
                    else: self.stable_counter += 1
            except: pass

        # Analyze if stable
        if self.stable_counter > 5 and self.stable_prediction != "neutral":
            if not self.locked_exercise: self.locked_exercise = self.stable_prediction
            self.analyze_frame(self.stable_prediction, landmarks)
        else:
            self.form_status = "Identifying..."
            
        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if not landmarks: return
        p = [self.Point(lm) for lm in landmarks]
        ls, rs, le, re, lw, rw = p[11], p[12], p[13], p[14], p[15], p[16]
        lh, rh, lk, rk, la, ra = p[23], p[24], p[25], p[26], p[27], p[28]

        now = time.time(); self.form_status = "CORRECT FORM"; prev_reps = self.rep_counter
        
        try:
            # --- UPDATED 3D LOGIC FOR ALL EXERCISES ---
            if exercise_name == 'bicepCurl':
                ang = min(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                swing = max(calculate_inclination_3d(ls, le), calculate_inclination_3d(rs, re))
                self.debug_angles = {"Elbow": int(ang), "Swing": int(swing)}
                
                if ang > 145: self.stage = "down"
                if ang < 70 and self.stage == "down":
                    if now - self.last_rep_time > 0.6: self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now
                
                # FIX FOR RED SKELETON: Relaxed Swing Threshold to 65
                if swing > 65: self.form_status = "ERROR: ELBOWS SWINGING"
                
                # FORCE GREEN if rep was just counted
                if (now - self.last_rep_time) < 1.0: self.form_status = "GOOD FORM"

            elif exercise_name == 'lateralRaise':
                sh = max(calculate_inclination_3d(ls, le), calculate_inclination_3d(rs, re))
                self.debug_angles = {"Shoulder": int(sh)}
                if sh < 35: self.stage = "down"
                if sh > 80 and self.stage == "down":
                    if now - self.last_rep_time > 0.6: self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now
                if sh > 115: self.form_status = "WARNING: TOO HIGH"

            elif exercise_name == 'shoulderPress':
                elb = max(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                self.debug_angles = {"Elbow": int(elb)}
                if elb < 100: self.stage = "down"
                if elb > 160 and self.stage == "down":
                    if now - self.last_rep_time > 0.6: self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now

            elif exercise_name in ['bentOverRow', 'dumbbellReverseFly']:
                torso = calculate_inclination_3d(ls, lh)
                elb = min(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                self.debug_angles = {"Torso": int(torso), "Elbow": int(elb)}
                if torso < 40: self.form_status = "ERROR: BEND OVER MORE"
                else:
                    if elb > 145: self.stage = "down"
                    if elb < 100 and self.stage == "down":
                        if now - self.last_rep_time > 0.6: self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now

            elif exercise_name == 'tricepKickback':
                uarm = max(calculate_inclination_3d(ls, le), calculate_inclination_3d(rs, re))
                elb = max(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                self.debug_angles = {"Arm": int(uarm), "Elbow": int(elb)}
                
                # Check 1: Upper Arm must be high (low inclination relative to horizontal)
                if uarm < 60: self.form_status = "ERROR: LIFT ELBOW HIGHER"
                else:
                    if elb < 100: self.stage = "down"
                    if elb > 150 and self.stage == "down":
                        if now - self.last_rep_time > 0.6: self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now

            elif exercise_name in ['squat', 'gobletSquat']:
                knee = (calculate_angle_3d(lh, lk, la) + calculate_angle_3d(rh, rk, ra)) / 2
                self.debug_angles = {"Knee": int(knee)}
                if knee > 165: self.stage = "up"
                if knee < 105 and self.stage == "up":
                    if now - self.last_rep_time > 0.8: self.rep_counter += 1; self.stage = "down"; self.last_rep_time = now
                if knee < 120: self.form_status = "GOOD DEPTH"

            # --- ERROR LOGGING SYSTEM ---
            if self.rep_counter > prev_reps:
                if "ERROR" in self.form_status or "WARNING" in self.form_status:
                    self.new_error_to_log = {"rep_number": self.rep_counter, "error_type": self.form_status, "exercise_name": exercise_name}
                    if self.form_status == self.last_consecutive_error_type: self.consecutive_error_counter += 1
                    else: self.last_consecutive_error_type = self.form_status; self.consecutive_error_counter = 1
                else:
                    self.consecutive_error_counter = 0; self.last_consecutive_error_type = None
                
                if self.consecutive_error_counter >= 6:
                    msg = f"Repeated Error: {self.last_consecutive_error_type.replace('ERROR: ', '')}"
                    self.triggered_alert = {'message': msg, 'exercise': exercise_name, 'reps': self.rep_counter}
                    self.consecutive_error_counter = 0
        except: pass

    # Utility functions
    def get_triggered_alert(self):
        alert = self.triggered_alert; self.triggered_alert = None; return alert
    def get_new_error_log(self):
        log = self.new_error_to_log; self.new_error_to_log = None; return log
    def reset_session(self):
        self.rep_counter = 0; self.stage = None; self.angle_sequence_buffer.clear(); self.locked_exercise = None
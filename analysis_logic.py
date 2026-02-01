import numpy as np
import time
import math
from collections import deque, Counter

# --- HELPER FUNCTIONS ---
def calculate_inclination(point_top, point_bottom):
    """Calculates angle relative to vertical gravity (0=Vertical, 90=Horizontal)"""
    p1_x = point_top['x'] if isinstance(point_top, dict) else point_top.x
    p1_y = point_top['y'] if isinstance(point_top, dict) else point_top.y
    p2_x = point_bottom['x'] if isinstance(point_bottom, dict) else point_bottom.x
    p2_y = point_bottom['y'] if isinstance(point_bottom, dict) else point_bottom.y
    dx = p2_x - p1_x; dy = p2_y - p1_y
    angle_rad = math.atan2(dy, dx)
    return abs(abs(math.degrees(angle_rad)) - 90)

def calculate_angle_2d(a, b, c):
    """Calculates inner angle at point b"""
    a = np.array(a[:2]); b = np.array(b[:2]); c = np.array(c[:2])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def normalize_pose_robust(landmarks):
    try:
        lms = []
        for lm in landmarks:
            if isinstance(lm, dict): lms.append([float(lm['x']), float(lm['y']), float(lm.get('z', 0.0))])
            else: lms.append([float(lm.x), float(lm.y), float(lm.z)])
        landmarks_np = np.array(lms, dtype=np.float32)
        hip_center_3d = (landmarks_np[23] + landmarks_np[24]) / 2.0
        left_shoulder = landmarks_np[11][:2]; right_shoulder = landmarks_np[12][:2]
        left_hip = landmarks_np[23][:2]; right_hip = landmarks_np[24][:2]
        torso_len = np.linalg.norm(((left_hip+right_hip)/2.0) - ((left_shoulder+right_shoulder)/2.0)) + 1e-6
        return (landmarks_np - hip_center_3d) / torso_len
    except: return None

def extract_engineered_features(landmarks):
    norm_lms = normalize_pose_robust(landmarks)
    if norm_lms is None: return None
    def lm(i): return norm_lms[i]
    def ang(a,b,c): return calculate_angle_2d(lm(a), lm(b), lm(c))
    def dist(i, j): return np.linalg.norm(lm(i) - lm(j))
    
    angles = [
        ang(11,23,25), ang(12,24,26), ang(11,24,12), ang(0,7,8), ang(23,11,12), ang(24,12,11),
        ang(11,13,15), ang(12,14,16), ang(23,11,13), ang(24,12,14), ang(13,15,19), ang(14,16,20),
        ang(11,23,25), ang(12,24,26), ang(23,25,27), ang(24,26,28), ang(25,27,29), ang(26,28,30),
        ang(12,23,24), ang(11,24,23)
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
    if len(features) == 42: features = np.concatenate([features, np.zeros(5, dtype=np.float32)])
    return features

class ExerciseAnalyzer:
    def __init__(self, sequence_length=90, conf_threshold=0.60, stability_frames=3, reset_timeout=5.0):
        self.rep_counter = 0; self.stage = None; self.form_status = "START EXERCISE"; self.status_color = (0, 255, 0)
        self.previous_exercise = "neutral"; self.last_rep_time = time.time(); self.RESET_TIMEOUT = reset_timeout
        self.debug_angles = {} 
        self.model_configured = False; self.expected_seq_len = int(sequence_length); self.input_size = 0
        self.angle_sequence_buffer = deque(maxlen=self.expected_seq_len)
        self.CONF_THRESHOLD = conf_threshold
        self.STABILITY_FRAMES = stability_frames 
        self.recent_predictions = deque(maxlen=self.STABILITY_FRAMES); self.stable_prediction = "neutral"
        self.locked_exercise = None; self.neutral_persistence_counter = 0 
        self.frame_count = 0; self.PREDICTION_INTERVAL = 2; self.stable_counter = 0 
        self.triggered_alert = None; self.consecutive_error_counter = 0; self.last_consecutive_error_type = None; self.new_error_to_log = None

    def _auto_configure_model(self, input_details):
        shape = input_details[0]['shape']; self.input_size = int(shape[-1])
        if len(shape) == 3: self.expected_seq_len = int(shape[1]); self.angle_sequence_buffer = deque(maxlen=self.expected_seq_len)
        self.model_configured = True

    def predict_with_tflite(self, interpreter, input_details, output_details, input_data):
        input_index = input_details[0]['index']
        if input_details[0]['dtype'] != np.float32:
            scale, zero_point = input_details[0]['quantization']
            if scale > 0:
                input_data = (input_data / scale) + zero_point
                input_data = np.clip(input_data, -128, 127) if input_details[0]['dtype'] == np.int8 else np.clip(input_data, 0, 255)
                input_data = input_data.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_index, input_data); interpreter.invoke(); return interpreter.get_tensor(output_details[0]['index'])[0]

    def _apply_logic_override(self, ai_prediction, landmarks):
        """
        Robust Priority Logic:
        1. Sanity Bounds (Head vs Hip levels).
        2. Geometry Rescues (If AI misses).
        3. Removed 'Bent Over' blockers to fix camera angle bugs.
        """
        if not landmarks: return ai_prediction
        
        class P:
            def __init__(self, obj):
                self.x = obj['x'] if isinstance(obj, dict) else obj.x
                self.y = obj['y'] if isinstance(obj, dict) else obj.y

        ls = P(landmarks[11]); rs = P(landmarks[12]) # Shoulders
        le = P(landmarks[13]); re = P(landmarks[14]) # Elbows
        lw = P(landmarks[15]); rw = P(landmarks[16]) # Wrists
        nose = P(landmarks[0])

        # Calculate Geometric States
        # 1. Hand Height Check (Y increases downwards!)
        # Hands above nose (Lowest Y value)
        hands_above_head = (lw.y < nose.y) or (rw.y < nose.y)
        # Hands below shoulders
        hands_below_shoulder = (lw.y > ls.y) and (rw.y > rs.y)

        # 2. Arm Flare (Shoulder-Elbow Angle)
        # 0 = Vertical (Arms down), 90 = Horizontal (T-Pose)
        l_arm_inc = calculate_inclination(ls, le)
        r_arm_inc = calculate_inclination(rs, re)
        avg_arm_flare = (l_arm_inc + r_arm_inc) / 2.0
        
        # 3. Elbow Bend (0 = Folded, 180 = Straight)
        l_elb = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
        r_elb = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
        avg_elb = (l_elb + r_elb) / 2.0


        # --- RULE 1: SANITY BOUNDS (Override AI Confusion) ---
        
        # If hands are ABOVE HEAD, it MUST be Shoulder Press (or jumping jack)
        # It cannot be Lateral Raise or Bicep Curl
        if hands_above_head:
            if ai_prediction == 'lateralRaise': return 'shoulderPress'
            if ai_prediction == 'bicepCurl': return 'shoulderPress'

        # --- RULE 2: GEOMETRY RESCUE (If AI says Neutral) ---
        if ai_prediction == "neutral":
            
            # RESCUE: LATERAL RAISE
            # Logic: Arms flared out (>45 deg) AND not fully straight up
            # We relax elbow constraint to 90 deg to catch bent-arm variations
            if avg_arm_flare > 45 and avg_elb > 90:
                if not hands_above_head: # Don't confuse with press top
                    return "lateralRaise"

            # RESCUE: SHOULDER PRESS
            # Logic: Hands above head. Period.
            if hands_above_head:
                return "shoulderPress"

            # RESCUE: BICEP CURL
            # Logic: One arm is bent (< 110) AND Wrist is higher than elbow
            # Check Left
            if l_elb < 110 and lw.y < le.y: return "bicepCurl"
            # Check Right
            if r_elb < 110 and rw.y < re.y: return "bicepCurl"

        return ai_prediction 

    def process_frame(self, interpreter, input_details, output_details, label_mapping, landmarks, current_exercise, scaler=None):
        self.frame_count += 1
        if not self.model_configured: self._auto_configure_model(input_details)
        if self.input_size == 47: 
            features = extract_engineered_features(landmarks)
            if features is not None and scaler is not None:
                try: features = scaler.transform(features.reshape(1, -1)).flatten()
                except Exception: pass
        else: features = np.zeros(self.input_size, dtype=np.float32)
        if features is None: return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles
        self.angle_sequence_buffer.append(features)
        
        if len(self.angle_sequence_buffer) == self.expected_seq_len and self.frame_count % self.PREDICTION_INTERVAL == 0:
            try:
                input_tensor = np.expand_dims(np.array(self.angle_sequence_buffer), axis=0)
                prediction = self.predict_with_tflite(interpreter, input_details, output_details, input_tensor.astype(np.float32))
                if np.max(prediction) > 1.0: prediction = np.exp(prediction - np.max(prediction)); prediction = prediction / prediction.sum()
                idx = int(np.argmax(prediction)); conf = prediction[idx]
                raw_label = str(label_mapping.get(idx, label_mapping.get(str(idx), "neutral")))
                
                # Apply the robust overrides
                final_label = self._apply_logic_override(raw_label, landmarks)
                
                # Locking Logic
                if self.locked_exercise:
                    if final_label == "neutral":
                        self.neutral_persistence_counter += 1
                        if self.neutral_persistence_counter > 60: 
                            self.locked_exercise = None; self.neutral_persistence_counter = 0; self.rep_counter = 0; final_label = "neutral"
                        else: final_label = self.locked_exercise
                    else: self.neutral_persistence_counter = 0; final_label = self.locked_exercise

                self.recent_predictions.append(final_label)
                most_common, count = Counter(self.recent_predictions).most_common(1)[0]
                
                if count > (self.STABILITY_FRAMES // 2): 
                    if self.stable_prediction == most_common: self.stable_counter += 1
                    else:
                        if not self.locked_exercise:
                            if most_common != "neutral" and self.stable_prediction != "neutral": self.rep_counter = 0; self.stage = None
                        self.stable_prediction = most_common; self.stable_counter = 0 
            except Exception: pass

        if self.stable_counter > 3 and self.stable_prediction != "neutral": 
             if not self.locked_exercise: self.locked_exercise = self.stable_prediction
             self.analyze_frame(self.stable_prediction, landmarks)
        elif self.stable_prediction != "neutral": self.form_status = f"Verifying {self.stable_prediction}..."
        else: self.form_status = "Identifying Exercise..."
        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if not landmarks: return
        class Point:
            def __init__(self, obj):
                self.x = float(obj['x']) if isinstance(obj, dict) else float(obj.x)
                self.y = float(obj['y']) if isinstance(obj, dict) else float(obj.y)
        lms = [Point(lm) for lm in landmarks]
        MIN_REP_DURATION = 0.5; # Fast reps allowed
        current_time = time.time(); self.form_status = "CORRECT FORM"; self.status_color = (0, 255, 0)
        prev_rep_counter = self.rep_counter
        
        try:
            ls, rs = lms[11], lms[12]; le, re = lms[13], lms[14]; lw, rw = lms[15], lms[16]
            lh, rh = lms[23], lms[24]; lk, rk = lms[25], lms[26]; la, ra = lms[27], lms[28]
            
            if exercise_name == 'bicepCurl':
                # Track best arm
                l_angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                r_angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                
                if l_angle < r_angle: angle = l_angle; active_side = "Left"
                else: angle = r_angle; active_side = "Right"

                self.debug_angles = {'Elbow': int(angle)}
                if angle > 140: self.stage = "down" 
                if angle < 60 and self.stage == 'down': 
                    if (current_time - self.last_rep_time) > MIN_REP_DURATION: self.stage = "up"; self.rep_counter += 1; self.last_rep_time = current_time
                
                if active_side == "Left": sh_angle = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                else: sh_angle = calculate_angle_2d([re.x, re.y], [rs.x, rs.y], [rh.x, rh.y])
                if sh_angle > 50: self.form_status = "ERROR: ELBOWS SWINGING"

            elif exercise_name == 'lateralRaise':
                l_sh_ang = calculate_angle_2d([le.x, le.y], [ls.x, ls.y], [lh.x, lh.y])
                r_sh_ang = calculate_angle_2d([re.x, re.y], [rs.x, rs.y], [rh.x, rh.y])
                max_ang = max(l_sh_ang, r_sh_ang)

                self.debug_angles = {'Shoulder': int(max_ang)}

                if max_ang < 45: self.stage = "down"
                if max_ang > 75 and self.stage == 'down': 
                    if (current_time - self.last_rep_time) > MIN_REP_DURATION: 
                        self.stage = "up"; self.rep_counter += 1; self.last_rep_time = current_time
                
                if max_ang > 110: self.form_status = "WARNING: TOO HIGH"

            elif exercise_name == 'shoulderPress':
                elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                self.debug_angles = {'Elbow': int(elb_ang)}
                if elb_ang < 90: self.stage = "down" 
                if elb_ang > 150 and self.stage == "down": 
                    if (current_time - self.last_rep_time) > MIN_REP_DURATION: 
                        self.stage = "up"; self.rep_counter += 1; self.last_rep_time = current_time

            elif exercise_name in ['dumbbellReverseFly', 'bentOverRow']:
                torso_inc = calculate_inclination(ls, lh)
                l_elb_ang = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                r_elb_ang = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                active_elb_ang = min(l_elb_ang, r_elb_ang) 

                self.debug_angles = {'Torso': int(torso_inc), 'Elbow': int(active_elb_ang)}
                
                if torso_inc < 30: self.form_status = "ERROR: BEND OVER MORE"
                else:
                    if active_elb_ang > 150: self.stage = "down" 
                    if active_elb_ang < 100 and self.stage == "down": 
                         if (current_time - self.last_rep_time) > MIN_REP_DURATION: 
                            self.stage = "up"; self.rep_counter += 1; self.last_rep_time = current_time

            elif exercise_name in ['squat', 'gobletSquat']:
                hip_below_knee = (lh.y > (lk.y * 0.95)) or (rh.y > (rk.y * 0.95))
                knee_ang = calculate_angle_2d([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])
                self.debug_angles = {'Knee': int(knee_ang), 'Depth': 'GOOD' if hip_below_knee else 'HIGH'}
                if not hip_below_knee and self.stage == 'down': self.form_status = "WARNING: GO LOWER"
                if knee_ang > 160: self.stage = "up"
                if hip_below_knee and self.stage == "up":
                    if (current_time - self.last_rep_time) > MIN_REP_DURATION: 
                        self.stage = "down"; self.rep_counter += 1; self.last_rep_time = current_time

            if (self.rep_counter > prev_rep_counter):
                if "ERROR" in self.form_status:
                    self.new_error_to_log = { "rep_number": self.rep_counter, "error_type": self.form_status, "exercise_name": exercise_name }
                    if self.form_status == self.last_consecutive_error_type: self.consecutive_error_counter += 1
                    else: self.last_consecutive_error_type = self.form_status; self.consecutive_error_counter = 1
                else: self.consecutive_error_counter = 0; self.last_consecutive_error_type = None
                if self.consecutive_error_counter >= 6:
                    msg = f"Repeated Error: {self.last_consecutive_error_type.replace('ERROR: ', '')}"
                    self.triggered_alert = {'message': msg, 'exercise': exercise_name, 'reps': self.rep_counter}; self.consecutive_error_counter = 0
        except Exception: pass

    def get_triggered_alert(self):
        alert = self.triggered_alert; self.triggered_alert = None; return alert
    def get_new_error_log(self):
        log = self.new_error_to_log; self.new_error_to_log = None; return log
    def reset_session(self):
        self.rep_counter = 0; self.stage = None; self.angle_sequence_buffer.clear(); self.locked_exercise = None
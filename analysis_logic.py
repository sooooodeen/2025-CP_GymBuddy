import numpy as np
import time
import math
from collections import deque, Counter


def calculate_angle_3d(a, b, c):
    """Calculates the 3D angle at point b."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_inclination_3d(point_top, point_bottom):
    """Calculates verticality (0=Vertical Standing, 90=Bent Over Horizontal)."""
    p1 = np.array([point_top.x, point_top.y, point_top.z])
    p2 = np.array([point_bottom.x, point_bottom.y, point_bottom.z])
    
    vector = p2 - p1 
    unit_vector = vector / (np.linalg.norm(vector) + 1e-7)
    
    dot_product = np.dot(unit_vector, [0, 1, 0]) 
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    return angle 

def normalize_pose_robust(landmarks):
    try:
        lms = []
        for lm in landmarks:
            if isinstance(lm, dict):
                lms.append([float(lm['x']), float(lm['y']), float(lm.get('z', 0.0))])
            else:
                lms.append([float(lm.x), float(lm.y), float(lm.z)])
        
        landmarks_np = np.array(lms, dtype=np.float32)
        hip_center_3d = (landmarks_np[23] + landmarks_np[24]) / 2.0
        torso_len = max(np.linalg.norm(landmarks_np[11] - landmarks_np[23]), 0.1)
        
        return (landmarks_np - hip_center_3d) / torso_len
    except:
        return None

def extract_engineered_features(landmarks):
    norm_lms = normalize_pose_robust(landmarks)
    if norm_lms is None: return None
    def lm(i): return norm_lms[i]
    
    def ang_2d(a,b,c):
        v1 = lm(a)[:2] - lm(b)[:2]; v2 = lm(c)[:2] - lm(b)[:2]
        res = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
        return abs(res) if abs(res) <= 180 else 360 - abs(res)
    def dist(i, j): return np.linalg.norm(lm(i) - lm(j))
    
    def ang_3d_feat(i, j, k):
        a, b, c = lm(i), lm(j), lm(k)
        ba = a - b; bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    angles = [
        ang_2d(11,23,25), ang_2d(12,24,26), ang_2d(11,24,12), ang_2d(0,7,8), 
        ang_2d(23,11,12), ang_2d(24,12,11), ang_2d(11,13,15), ang_2d(12,14,16), 
        ang_2d(23,11,13), ang_2d(24,12,14), ang_2d(13,15,19), ang_2d(14,16,20),
        ang_2d(23,25,27), ang_2d(24,26,28), ang_2d(25,27,29), ang_2d(26,28,30),
        ang_2d(12,23,24), ang_2d(11,24,23), ang_3d_feat(23,25,27), ang_3d_feat(24,26,28),
        ang_3d_feat(11,23,25), ang_3d_feat(12,24,26)
    ]
    
    hip_center = np.array([0.0, 0.0, 0.0])
    
    distances = [
        dist(11, 12), dist(23, 24), dist(15, 25), dist(16, 26), dist(13, 23), dist(14, 24), 
        dist(27, 15), dist(28, 16),
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
    target_len = 47
    if len(features) < target_len: 
        features = np.concatenate([features, np.zeros(target_len - len(features))])
    return features


class ExerciseAnalyzer:
    def __init__(self, sequence_length=45, conf_threshold=0.60, stability_frames=5, reset_timeout=6.0):
        self.rep_counter = 0
        self.stage = None
        self.form_status = "START EXERCISE"
        self.last_rep_time = time.time()
        self.RESET_TIMEOUT = reset_timeout
        self.model_configured = False
        self.expected_seq_len = int(sequence_length)
        self.input_size = 0
        self.angle_sequence_buffer = deque(maxlen=self.expected_seq_len)
        self.CONF_THRESHOLD = conf_threshold
        self.STABILITY_FRAMES = stability_frames
        self.recent_predictions = deque(maxlen=self.STABILITY_FRAMES)
        self.stable_prediction = "neutral"
        self.locked_exercise = None
        self.neutral_persistence_counter = 0
        self.frame_count = 0
        self.PREDICTION_INTERVAL = 2
        self.stable_counter = 0
        self.triggered_alert = None
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.new_error_to_log = None
        self.debug_angles = {}

        self.ALLOWED_EXERCISES = {
            "bicepCurl", "shoulderPress", "lateralRaise", "bentOverRow", "uprightRow"
        }

    class Point:
        def __init__(self, lm):
            self.x = lm['x'] if isinstance(lm, dict) else lm.x
            self.y = lm['y'] if isinstance(lm, dict) else lm.y
            self.z = lm['z'] if isinstance(lm, dict) else lm.z

    def _auto_configure_model(self, input_details):
        shape = input_details[0]['shape']
        self.input_size = int(shape[-1])
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
        if not landmarks: return ai_prediction
        p = [self.Point(lm) for lm in landmarks]
        
        ls, rs = p[11], p[12] 
        le, re = p[13], p[14] 
        lw, rw = p[15], p[16] 
        lh, rh = p[23], p[24] 
        
        sh_width = max(abs(ls.x - rs.x), 0.01) 
        hand_span = abs(lw.x - rw.x)
        elbow_span = abs(le.x - re.x) 
        
        torso_height_y = abs(lh.y - ls.y)
        mid_chest_y = ls.y + (torso_height_y * 0.45) 

        if ai_prediction not in self.ALLOWED_EXERCISES and ai_prediction != "neutral":
            if ai_prediction == "romanianDeadlift":
                ai_prediction = "bentOverRow"
            else:
                ai_prediction = "neutral"

        if (lw.y < ls.y) and (rw.y < rs.y):
            return "shoulderPress"

        if hand_span > (sh_width * 1.6) and (lw.y < lh.y):
            return "lateralRaise"

        if (lw.y < lh.y - 0.05) and (rw.y < rh.y - 0.05):
            
            if (lw.y < le.y - 0.02) and (rw.y < re.y - 0.02):
                return "bicepCurl"
                
            if (le.y <= lw.y) and (re.y <= rw.y):
                
                if (lw.y < mid_chest_y) and (rw.y < mid_chest_y):
                    return "uprightRow"
                else:
                    return "bentOverRow"

        if ai_prediction == "uprightRow":
            if elbow_span < (sh_width * 1.1):
                return "bicepCurl" 

        if ai_prediction == "bicepCurl":
            if elbow_span > (sh_width * 1.3):
                return "uprightRow"

        if ai_prediction not in self.ALLOWED_EXERCISES:
            return "neutral"

        return ai_prediction

    def process_frame(self, interpreter, input_details, output_details, label_mapping, landmarks, current_exercise, scaler=None):
        self.frame_count += 1
        if not self.model_configured: self._auto_configure_model(input_details)
        
        features = extract_engineered_features(landmarks) 
        if features is None: return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles
        if scaler:
            try: features = scaler.transform(features.reshape(1, -1)).flatten()
            except: pass
        self.angle_sequence_buffer.append(features)

        if len(self.angle_sequence_buffer) == self.expected_seq_len and self.frame_count % self.PREDICTION_INTERVAL == 0:
            try:
                input_tensor = np.expand_dims(np.array(self.angle_sequence_buffer), axis=0).astype(np.float32)
                prediction = self.predict_with_tflite(interpreter, input_details, output_details, input_tensor)
                idx = int(np.argmax(prediction))
                raw_label = str(label_mapping.get(idx, "neutral"))
                final_label = self._apply_logic_override(raw_label, landmarks)
                
                if self.locked_exercise:
                    if final_label == "neutral":
                        self.neutral_persistence_counter += 1
                        if self.neutral_persistence_counter > 40:
                            self.locked_exercise = None; self.rep_counter = 0; self.stage = None; final_label = "neutral"
                        else: final_label = self.locked_exercise
                    else:
                        self.neutral_persistence_counter = 0
                        final_label = self.locked_exercise

                self.recent_predictions.append(final_label)
                most_common, count = Counter(self.recent_predictions).most_common(1)[0]
                
                if count >= 3:
                    if self.stable_prediction != most_common:
                        if not self.locked_exercise: self.rep_counter = 0; self.stage = None
                        self.stable_prediction = most_common; self.stable_counter = 0
                    else: self.stable_counter += 1
            except Exception as e: print(f"Prediction Error: {e}")

        if self.stable_counter > 5 and self.stable_prediction != "neutral":
            if not self.locked_exercise: self.locked_exercise = self.stable_prediction
            self.analyze_frame(self.stable_prediction, landmarks)
        elif not self.locked_exercise:
            self.form_status = "Identifying..."
            
        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if not landmarks: return
        p = [self.Point(lm) for lm in landmarks]
        ls, rs, le, re, lw, rw = p[11], p[12], p[13], p[14], p[15], p[16]
        lh, rh = p[23], p[24]

        now = time.time()
        self.form_status = "CORRECT FORM"
        prev_reps = self.rep_counter
        
        try:
            # --- 1. BICEP CURL ---
            if exercise_name == 'bicepCurl':
                ang = min(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                l_swing = calculate_angle_3d(le, ls, lh)
                r_swing = calculate_angle_3d(re, rs, rh)
                swing = max(l_swing, r_swing)
                self.debug_angles = {"Elbow": int(ang), "Swing": int(swing)}
                
                if ang > 150: self.stage = "down"
                if ang < 85 and self.stage == "down":
                    vertical_alignment = abs(lw.x - ls.x) 
                    if vertical_alignment < 0.15: 
                        if now - self.last_rep_time > 0.6: 
                            self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now
                if swing > 45: self.form_status = "ERROR: ELBOWS SWINGING"

            # --- 2. LATERAL RAISE ---
            elif exercise_name == 'lateralRaise':
                l_height = calculate_angle_3d(le, ls, lh)
                r_height = calculate_angle_3d(re, rs, rh)
                sh = max(l_height, r_height)
                self.debug_angles = {"Shoulder": int(sh)}
                
                if sh < 35: self.stage = "down"
                if sh > 80 and self.stage == "down":
                    if now - self.last_rep_time > 0.6: 
                        self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now
                if sh > 160: self.form_status = "WARNING: TOO HIGH"

            # --- 3. SHOULDER PRESS ---
            elif exercise_name == 'shoulderPress':
                elb = max(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                self.debug_angles = {"Elbow": int(elb)}
                if elb < 90: self.stage = "down"
                if elb > 155 and self.stage == "down":
                    if now - self.last_rep_time > 0.6: 
                        self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now

            # --- 4. BENT OVER ROW ---
            elif exercise_name == 'bentOverRow': 
                elb = min(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                
                torso_height_y = abs(lh.y - ls.y)
                mid_chest_y = ls.y + (torso_height_y * 0.45)
                
                self.debug_angles = {"Elbow": int(elb)}
                
                if (lw.y < mid_chest_y) and (rw.y < mid_chest_y): 
                    self.form_status = "ERROR: PULLING TOO HIGH"
                else:
                    if elb > 150: self.stage = "down"
                    if elb < 100 and self.stage == "down":
                        if now - self.last_rep_time > 0.6: 
                            self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now

            # --- 5. UPRIGHT ROW ---
            elif exercise_name == 'uprightRow':
                l_hand_height = (lh.y - lw.y) 
                r_hand_height = (rh.y - rw.y)
                avg_h = (l_hand_height + r_hand_height) / 2.0
                
                self.debug_angles = {"Hand Height": round(avg_h, 2)}
                
                if avg_h < 0.1: self.stage = "down" 
                
                if avg_h > 0.25 and self.stage == "down":
                     if now - self.last_rep_time > 0.6: 
                        self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now
                
                if self.stage == "up":
                    if (le.y > lw.y + 0.05) or (re.y > rw.y + 0.05):
                        self.form_status = "ERROR: ELBOWS LOWER THAN WRISTS"
                    else:
                        self.form_status = "CORRECT FORM"
                else:
                    self.form_status = "CORRECT FORM"

            if self.rep_counter > prev_reps:
                if "ERROR" in self.form_status or "WARNING" in self.form_status:
                    self.new_error_to_log = {"rep_number": self.rep_counter, "error_type": self.form_status, "exercise_name": exercise_name}
                    if self.form_status == self.last_consecutive_error_type: 
                        self.consecutive_error_counter += 1
                    else: 
                        self.last_consecutive_error_type = self.form_status; self.consecutive_error_counter = 1
                else:
                    self.consecutive_error_counter = 0; self.last_consecutive_error_type = None
                
                if self.consecutive_error_counter >= 4:
                    msg = f"Repeated Error: {self.last_consecutive_error_type.replace('ERROR: ', '')}"
                    self.triggered_alert = {'message': msg, 'exercise': exercise_name, 'reps': self.rep_counter}
                    self.consecutive_error_counter = 0
        except Exception as e:
            print(f"Analysis Error: {e}")

    def get_triggered_alert(self):
        alert = self.triggered_alert; self.triggered_alert = None; return alert
    def get_new_error_log(self):
        log = self.new_error_to_log; self.new_error_to_log = None; return log
    def reset_session(self):
        self.rep_counter = 0; self.stage = None; self.angle_sequence_buffer.clear(); self.locked_exercise = None
        # Lateral raise = Knee Level
        # Bicep Curl = Half body or Knee Level
        # Shoulder Pres = Show Higher part for arm / Knee Level
        # Upright Row = Below Knee Level 
        # Bent Over Row = Half Body or Knee Level
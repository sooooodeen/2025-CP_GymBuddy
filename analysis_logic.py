import numpy as np
import time
import math
from collections import deque, Counter

# --- 1. GEOMETRY ENGINE ---

def calculate_angle_3d(a, b, c):
    """Calculates the 3D angle at point b."""
    # Ensure inputs are numpy arrays for vector math
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    ba = a - b
    bc = c - b
    
    # Calculate cosine similarity
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_inclination_3d(point_top, point_bottom):
    """Calculates verticality (0=Vertical Standing, 90=Bent Over Horizontal)."""
    p1 = np.array([point_top.x, point_top.y, point_top.z])
    p2 = np.array([point_bottom.x, point_bottom.y, point_bottom.z])
    
    vector = p2 - p1 # Vector points down from shoulder to hip
    unit_vector = vector / (np.linalg.norm(vector) + 1e-7)
    
    # Compare against pure vertical UP [0, 1, 0] (In Image coords, Y is down, so we check alignment)
    # Using standard [0, 1, 0] for vertical Y-axis calculation
    dot_product = np.dot(unit_vector, [0, 1, 0]) 
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    return abs(angle - 180)

# --- 2. DATA PREPROCESSING ---

def normalize_pose_robust(landmarks):
    try:
        lms = []
        for lm in landmarks:
            if isinstance(lm, dict):
                lms.append([float(lm['x']), float(lm['y']), float(lm.get('z', 0.0))])
            else:
                lms.append([float(lm.x), float(lm.y), float(lm.z)])
        
        landmarks_np = np.array(lms, dtype=np.float32)
        
        # Center pose at Hip Center
        hip_center_3d = (landmarks_np[23] + landmarks_np[24]) / 2.0
        
        # --- FIX: Robust Torso Length ---
        # Added max() to prevent divide by zero or negative length issues during weird angles
        torso_len = max(np.linalg.norm(landmarks_np[11] - landmarks_np[23]), 0.1)
        
        return (landmarks_np - hip_center_3d) / torso_len
    except:
        return None

def extract_engineered_features(landmarks):
    norm_lms = normalize_pose_robust(landmarks)
    if norm_lms is None: return None
    
    def lm(i): return norm_lms[i]
    
    def ang_2d(a,b,c):
        # 2D Angle calculation for input features
        v1 = lm(a)[:2] - lm(b)[:2]
        v2 = lm(c)[:2] - lm(b)[:2]
        res = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
        return abs(res) if abs(res) <= 180 else 360 - abs(res)
        
    def dist(i, j): return np.linalg.norm(lm(i) - lm(j))
    
    # --- FIX: Removed Duplicates ---
    # The previous code had "11,23,25" and others twice. Cleaned list:
    angles = [
        ang_2d(11,23,25), ang_2d(12,24,26), ang_2d(11,24,12), ang_2d(0,7,8), 
        ang_2d(23,11,12), ang_2d(24,12,11), ang_2d(11,13,15), ang_2d(12,14,16), 
        ang_2d(23,11,13), ang_2d(24,12,14), ang_2d(13,15,19), ang_2d(14,16,20),
        # Removed duplicates here
        ang_2d(23,25,27), ang_2d(24,26,28), ang_2d(25,27,29), ang_2d(26,28,30),
        ang_2d(12,23,24), ang_2d(11,24,23)
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
    
    # Padding to match model input size (usually 47 or 48)
    target_len = 47
    if len(features) < target_len: 
        features = np.concatenate([features, np.zeros(target_len - len(features))])
    elif len(features) > target_len:
        features = features[:target_len]
        
    return features

# --- 3. EXERCISE ANALYZER ---

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
        
        # Alerting System
        self.triggered_alert = None
        self.consecutive_error_counter = 0
        self.last_consecutive_error_type = None
        self.new_error_to_log = None
        self.debug_angles = {}

        # --- FIX: State Variables for Hysteresis ---
        self.is_bent_over = False  # Memory for rowing/kickbacks
        self.is_hands_up = False   # Memory for presses

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

    def _apply_logic_override(self, ai_prediction, landmarks, conf):
        """
        Prevents Impossible Moves and uses Hysteresis (Memory) to stop flickering.
        """
        if not landmarks: return ai_prediction
        
        p = [self.Point(lm) for lm in landmarks]
        ls, rs = p[11], p[12] # Shoulders
        lw, rw = p[15], p[16] # Wrists
        lh, rh = p[23], p[24] # Hips
        
        # Calculate Current States
        torso_inc = calculate_inclination_3d(ls, lh)
        
        # --- FIX 1: Hysteresis for Bent-Over Exercises ---
        # Logic: Require deep bend to enter state, but allow slight rise to stay in state.
        if torso_inc > 35: 
            self.is_bent_over = True
        elif torso_inc < 20: 
            self.is_bent_over = False
            
        # If AI says Row/Kickback but we are standing upright -> Force Neutral
        if ai_prediction in ["tricepKickback", "bentOverRow", "dumbbellReverseFly"]:
            if not self.is_bent_over:
                # Debug print if needed: print(f"Override: Too Upright ({int(torso_inc)})")
                return "neutral"

        # --- FIX 2: Shoulder Press Guard (Relaxed) ---
        # Logic: Hands must be near or above shoulders.
        # Added +0.15 buffer: Allows hands to dip slightly below shoulders at bottom of rep.
        hands_above_shoulders = (lw.y < ls.y + 0.15) or (rw.y < rs.y + 0.15)
        
        if ai_prediction == "shoulderPress" and not hands_above_shoulders:
            return "neutral"

        # --- FIX 3: Wide Arm Guard (Lateral Raise vs Curl) ---
        # Logic: If arms are very wide, it's likely a Raise, not a Curl.
        shoulder_width = abs(ls.x - rs.x)
        hand_width = abs(lw.x - rw.x)
        is_wide = hand_width > (shoulder_width * 1.6)
        
        if ai_prediction == "bicepCurl" and is_wide:
            return "lateralRaise"

        return ai_prediction

    def process_frame(self, interpreter, input_details, output_details, label_mapping, landmarks, current_exercise, scaler=None):
        self.frame_count += 1
        if not self.model_configured: self._auto_configure_model(input_details)
        
        features = extract_engineered_features(landmarks) 
        if features is None: return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles
        
        # Scaling (if scaler provided)
        if scaler:
            try: features = scaler.transform(features.reshape(1, -1)).flatten()
            except: pass
            
        self.angle_sequence_buffer.append(features)

        # Run Prediction Loop
        if len(self.angle_sequence_buffer) == self.expected_seq_len and self.frame_count % self.PREDICTION_INTERVAL == 0:
            try:
                input_tensor = np.expand_dims(np.array(self.angle_sequence_buffer), axis=0).astype(np.float32)
                prediction = self.predict_with_tflite(interpreter, input_details, output_details, input_tensor)
                
                idx = int(np.argmax(prediction))
                conf = prediction[idx]
                raw_label = str(label_mapping.get(idx, "neutral"))
                
                # Apply Logic Override (Fixing the flicker)
                final_label = self._apply_logic_override(raw_label, landmarks, conf)
                
                # Locking Logic (Sticky Exercise)
                if self.locked_exercise:
                    if final_label == "neutral":
                        self.neutral_persistence_counter += 1
                        # Wait 40 frames (~2 sec) before unlocking
                        if self.neutral_persistence_counter > 40:
                            self.locked_exercise = None
                            self.rep_counter = 0
                            self.stage = None
                            final_label = "neutral"
                        else:
                            final_label = self.locked_exercise
                    else:
                        self.neutral_persistence_counter = 0
                        # If AI switches to a different valid exercise, trust it if strong
                        if final_label != self.locked_exercise and conf > 0.85:
                             pass # Allow switch
                        else:
                             final_label = self.locked_exercise

                self.recent_predictions.append(final_label)
                most_common, count = Counter(self.recent_predictions).most_common(1)[0]
                
                # Stability Check
                if count >= 3: # Fast response
                    if self.stable_prediction != most_common:
                        if not self.locked_exercise: 
                            self.rep_counter = 0
                            self.stage = None
                        self.stable_prediction = most_common
                        self.stable_counter = 0
                    else:
                        self.stable_counter += 1
            except Exception as e:
                print(f"Prediction Error: {e}")

        # If stable for 5+ frames, lock and analyze
        if self.stable_counter > 5 and self.stable_prediction != "neutral":
            if not self.locked_exercise: 
                self.locked_exercise = self.stable_prediction
            self.analyze_frame(self.stable_prediction, landmarks)
        elif not self.locked_exercise:
            self.form_status = "Identifying..."
            
        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if not landmarks: return
        p = [self.Point(lm) for lm in landmarks]
        ls, rs, le, re, lw, rw = p[11], p[12], p[13], p[14], p[15], p[16]
        lh, rh, lk, rk, la, ra = p[23], p[24], p[25], p[26], p[27], p[28]

        now = time.time()
        self.form_status = "CORRECT FORM"
        prev_reps = self.rep_counter
        
        try:
            # --- BICEP CURL ---
            if exercise_name == 'bicepCurl':
                ang = min(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                swing = max(calculate_inclination_3d(ls, le), calculate_inclination_3d(rs, re))
                self.debug_angles = {"Elbow": int(ang), "Swing": int(swing)}
                
                if ang > 150: self.stage = "down"
                if ang < 85 and self.stage == "down": # Relaxed Up Threshold
                    if now - self.last_rep_time > 0.6: 
                        self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now
                if swing > 65: self.form_status = "ERROR: ELBOWS SWINGING"

            # --- LATERAL RAISE ---
            elif exercise_name == 'lateralRaise':
                sh = max(calculate_inclination_3d(ls, le), calculate_inclination_3d(rs, re))
                self.debug_angles = {"Shoulder": int(sh)}
                if sh < 40: self.stage = "down"
                if sh > 80 and self.stage == "down":
                    if now - self.last_rep_time > 0.6: 
                        self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now
                if sh > 110: self.form_status = "WARNING: TOO HIGH"

            # --- SHOULDER PRESS ---
            elif exercise_name == 'shoulderPress':
                elb = max(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                self.debug_angles = {"Elbow": int(elb)}
                if elb < 90: self.stage = "down"
                if elb > 155 and self.stage == "down":
                    if now - self.last_rep_time > 0.6: 
                        self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now

            # --- BENT OVER ROW / FLY ---
            elif exercise_name in ['bentOverRow', 'dumbbellReverseFly']:
                torso = calculate_inclination_3d(ls, lh)
                elb = min(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                self.debug_angles = {"Torso": int(torso), "Elbow": int(elb)}
                
                if torso < 25: self.form_status = "ERROR: BEND OVER MORE"
                else:
                    if elb > 150: self.stage = "down"
                    if elb < 100 and self.stage == "down":
                        if now - self.last_rep_time > 0.6: 
                            self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now

            # --- KICKBACK ---
            elif exercise_name == 'tricepKickback':
                uarm = max(calculate_inclination_3d(ls, le), calculate_inclination_3d(rs, re))
                elb = max(calculate_angle_3d(ls, le, lw), calculate_angle_3d(rs, re, rw))
                self.debug_angles = {"Arm": int(uarm), "Elbow": int(elb)}
                
                if uarm < 60: self.form_status = "ERROR: LIFT ELBOW HIGHER"
                else:
                    if elb < 90: self.stage = "down"
                    if elb > 150 and self.stage == "down":
                        if now - self.last_rep_time > 0.6: 
                            self.rep_counter += 1; self.stage = "up"; self.last_rep_time = now

            # --- SQUATS ---
            elif exercise_name in ['squat', 'gobletSquat']:
                # Averaging both knees for stability
                knee = (calculate_angle_3d(lh, lk, la) + calculate_angle_3d(rh, rk, ra)) / 2
                self.debug_angles = {"Knee": int(knee)}
                
                if knee > 160: self.stage = "up"
                if knee < 110 and self.stage == "up":
                    if now - self.last_rep_time > 0.8: 
                        self.rep_counter += 1; self.stage = "down"; self.last_rep_time = now
                if knee < 120: self.form_status = "GOOD DEPTH"

            # --- ERROR LOGGING ---
            if self.rep_counter > prev_reps:
                if "ERROR" in self.form_status or "WARNING" in self.form_status:
                    self.new_error_to_log = {"rep_number": self.rep_counter, "error_type": self.form_status, "exercise_name": exercise_name}
                    if self.form_status == self.last_consecutive_error_type: 
                        self.consecutive_error_counter += 1
                    else: 
                        self.last_consecutive_error_type = self.form_status; self.consecutive_error_counter = 1
                else:
                    self.consecutive_error_counter = 0; self.last_consecutive_error_type = None
                
                if self.consecutive_error_counter >= 4: # Reduced from 6 for faster feedback
                    msg = f"Repeated Error: {self.last_consecutive_error_type.replace('ERROR: ', '')}"
                    self.triggered_alert = {'message': msg, 'exercise': exercise_name, 'reps': self.rep_counter}
                    self.consecutive_error_counter = 0
        except Exception as e:
            print(f"Analysis Error: {e}")

    # Utility functions
    def get_triggered_alert(self):
        alert = self.triggered_alert; self.triggered_alert = None; return alert
    def get_new_error_log(self):
        log = self.new_error_to_log; self.new_error_to_log = None; return log
    def reset_session(self):
        self.rep_counter = 0; self.stage = None; self.angle_sequence_buffer.clear(); self.locked_exercise = None
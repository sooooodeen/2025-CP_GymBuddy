import numpy as np
import time
import math
from collections import deque, Counter

# --- 1. GEOMETRY & NORMALIZATION (Matches feature_engineering.py) ---

def normalize_pose_robust(landmarks):
    """
    Normalizes landmarks based on torso length (Hip Center to Shoulder Center).
    Matches the logic used in your training data preparation.
    """
    try:
        # Convert list of objects/dicts to numpy array [x, y, z]
        lms = []
        for lm in landmarks:
            # Handle both MediaPipe objects (lm.x) and dicts (lm['x'])
            if isinstance(lm, dict):
                lms.append([lm['x'], lm['y'], lm.get('z', 0.0)])
            else:
                lms.append([lm.x, lm.y, lm.z])
        
        landmarks_np = np.array(lms)

        # Indices: Left Hip=23, Right Hip=24, Left Shoulder=11, Right Shoulder=12
        left_hip = landmarks_np[23]
        right_hip = landmarks_np[24]
        hip_center = (left_hip + right_hip) / 2.0

        left_shoulder = landmarks_np[11]
        right_shoulder = landmarks_np[12]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        torso_length = np.linalg.norm(hip_center - shoulder_center) + 1e-6
        
        # Prevent division by zero if person is not fully visible
        if torso_length < 1e-5: return None

        # Normalize: Center hips at (0,0,0) and scale by torso length
        normalized_landmarks = (landmarks_np - hip_center) / torso_length
        return normalized_landmarks
    except Exception as e:
        print(f"Normalization Error: {e}")
        return None

def calculate_angle_3d(a, b, c):
    """Calculates 3D angle between points a, b, c (center at b)."""
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
    """2D Angle helper for the Rule-Based Form Checker."""
    a = np.array(a[:2]); b = np.array(b[:2]); c = np.array(c[:2])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 2. FEATURE ENGINEERING (The Fix for 47 Inputs) ---

def extract_engineered_features(landmarks):
    """
    Extracts the 42 features from feature_engineering.py + 5 pads = 47 features.
    """
    norm_lms = normalize_pose_robust(landmarks)
    if norm_lms is None: return None

    # Helper to get coord by index
    def lm(i): return norm_lms[i]

    # --- 1. ANGLES (20 Features) ---
    # Indices: 0=Nose, 7=L.Ear, 8=R.Ear, 11=L.Shoulder, 12=R.Shoulder, 
    # 13=L.Elbow, 14=R.Elbow, 15=L.Wrist, 16=R.Wrist, 
    # 23=L.Hip, 24=R.Hip, 25=L.Knee, 26=R.Knee, 27=L.Ankle, 28=R.Ankle
    
    angles = [
        # Torso & Neck
        calculate_angle_3d(lm(11), lm(23), lm(25)), # angle_torso_side_left
        calculate_angle_3d(lm(12), lm(24), lm(26)), # angle_torso_side_right
        calculate_angle_3d(lm(11), lm(24), lm(12)), # angle_torso_front
        calculate_angle_3d(lm(0), lm(7), lm(8)),    # angle_neck
        calculate_angle_3d(lm(23), lm(11), lm(12)), # angle_spine_hip_shoulder_left
        calculate_angle_3d(lm(24), lm(12), lm(11)), # angle_spine_hip_shoulder_right
        
        # Arms
        calculate_angle_3d(lm(11), lm(13), lm(15)), # angle_left_elbow
        calculate_angle_3d(lm(12), lm(14), lm(16)), # angle_right_elbow
        calculate_angle_3d(lm(23), lm(11), lm(13)), # angle_left_shoulder_abduction
        calculate_angle_3d(lm(24), lm(12), lm(14)), # angle_right_shoulder_abduction
        calculate_angle_3d(lm(13), lm(15), lm(19)), # angle_left_wrist (19=Index)
        calculate_angle_3d(lm(14), lm(16), lm(20)), # angle_right_wrist (20=Index)

        # Legs & Hips
        calculate_angle_3d(lm(11), lm(23), lm(25)), # angle_left_hip
        calculate_angle_3d(lm(12), lm(24), lm(26)), # angle_right_hip
        calculate_angle_3d(lm(23), lm(25), lm(27)), # angle_left_knee
        calculate_angle_3d(lm(24), lm(26), lm(28)), # angle_right_knee
        calculate_angle_3d(lm(25), lm(27), lm(29)), # angle_left_ankle (29=Heel)
        calculate_angle_3d(lm(26), lm(28), lm(30)), # angle_right_ankle (30=Heel)
        
        # Twist
        calculate_angle_3d(lm(12), lm(23), lm(24)), # angle_shoulder_hip_twist_left
        calculate_angle_3d(lm(11), lm(24), lm(23))  # angle_shoulder_hip_twist_right
    ]

    # --- 2. DISTANCES (22 Features) ---
    def dist(i, j): return np.linalg.norm(lm(i) - lm(j))
    
    # Hip Center is origin (0,0,0) in normalized space
    hip_center = np.array([0.0, 0.0, 0.0]) 

    distances = [
        dist(11, 12), # dist_shoulders
        dist(23, 24), # dist_hips
        dist(15, 25), # dist_left_wrist_knee
        dist(16, 26), # dist_right_wrist_knee
        dist(13, 23), # dist_left_elbow_hip
        dist(14, 24), # dist_right_elbow_hip
        dist(27, 15), # dist_left_ankle_wrist
        dist(28, 16), # dist_right_ankle_wrist
        np.linalg.norm(lm(0) - hip_center), # dist_nose_hip
        
        # Y-Distances (Vertical)
        abs(lm(15)[1] - lm(11)[1]), # dist_y_l_wrist_shoulder
        abs(lm(16)[1] - lm(12)[1]), # dist_y_r_wrist_shoulder
        abs(lm(23)[1] - lm(25)[1]), # dist_y_l_hip_knee
        abs(lm(24)[1] - lm(26)[1]), # dist_y_r_hip_knee
        abs(lm(11)[1] - lm(23)[1]), # dist_y_l_shoulder_hip
        abs(lm(12)[1] - lm(24)[1]), # dist_y_r_shoulder_hip
        abs(lm(27)[1] - lm(29)[1]), # dist_y_l_ankle_heel
        abs(lm(28)[1] - lm(30)[1]), # dist_y_r_ankle_heel

        # Z-Distances (Depth)
        abs(lm(15)[2] - lm(23)[2]), # dist_z_l_wrist_hip
        abs(lm(16)[2] - lm(24)[2]), # dist_z_r_wrist_hip
        abs(lm(11)[2] - lm(23)[2]), # dist_z_l_shoulder_hip
        abs(lm(12)[2] - lm(24)[2]), # dist_z_r_shoulder_hip
        abs(lm(0)[2] - hip_center[2]) # dist_z_nose_hip
    ]

    # Combine all features (Total 42)
    features = np.array(angles + distances, dtype=np.float32)
    
    # --- PADDING TO MATCH MODEL INPUT (47) ---
    # The logs showed [1, 90, 47]. We have 42 features. We add 5 zeros.
    if len(features) == 42:
        features = np.pad(features, (0, 5), 'constant')
        
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
        
        # Auto-Configuration State
        self.model_configured = False
        self.expected_seq_len = sequence_length
        self.input_size = 0
        
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
        """Detects input shape and configures extractor."""
        shape = input_details[0]['shape']
        self.input_size = shape[-1]
        
        print(f"\n--- [DEBUG] MODEL CONFIGURATION ---")
        print(f"Input Shape: {shape}")
        print(f"Expected Features: {self.input_size}")
        
        if len(shape) == 3:
            self.expected_seq_len = shape[1]
            self.angle_sequence_buffer = deque(maxlen=self.expected_seq_len)
            print(f"Sequence Length Set To: {self.expected_seq_len}")
        
        self.model_configured = True

    def predict_with_tflite(self, interpreter, input_details, output_details, input_data):
        input_index = input_details[0]['index']
        input_dtype = input_details[0]['dtype']
        
        # Auto-Quantization (Float -> Int)
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
        
        # Auto-Dequantization (Int -> Float)
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

        # 1. Extract Features (Adaptive)
        if self.input_size == 47:
            # Use the robust Feature Engineering (42 + 5 padding)
            features = extract_engineered_features(landmarks)
        elif self.input_size == 132:
            # Raw Landmarks (Fallback for other models)
            lms = []
            for lm in landmarks:
                x = lm['x'] if isinstance(lm, dict) else lm.x
                y = lm['y'] if isinstance(lm, dict) else lm.y
                z = lm.get('z', 0.0) if isinstance(lm, dict) else lm.z
                v = lm.get('visibility', 0.0) if isinstance(lm, dict) else lm.visibility
                lms.extend([x, y, z, v])
            features = np.array(lms, dtype=np.float32)
        else:
            # Fallback (all zeros) to prevent crash if unknown model
            features = np.zeros(self.input_size, dtype=np.float32)

        if features is None: 
            return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

        # 2. Update Buffer
        self.angle_sequence_buffer.append(features)
        
        # 3. Predict
        if len(self.angle_sequence_buffer) == self.expected_seq_len and self.frame_count % self.PREDICTION_INTERVAL == 0:
            try:
                input_tensor = np.expand_dims(np.array(self.angle_sequence_buffer), axis=0)
                
                prediction_output = self.predict_with_tflite(
                    interpreter, input_details, output_details, input_tensor.astype(np.float32)
                )

                # Softmax (if outputs are raw logits)
                if np.max(prediction_output) > 1.0 or np.min(prediction_output) < 0.0:
                     exp_x = np.exp(prediction_output - np.max(prediction_output))
                     prediction_output = exp_x / exp_x.sum()

                predicted_idx = np.argmax(prediction_output)
                confidence = prediction_output[predicted_idx]
                
                # Safe Lookup (Handles string keys in JSON map)
                safe_idx = int(predicted_idx)
                # Try int key first, then string key
                pred_label = label_mapping.get(safe_idx, label_mapping.get(str(safe_idx), "neutral"))

                # Debug Print (Visible in Terminal)
                # print(f"AI Pred: {pred_label} ({confidence:.2f})")

                if confidence > self.CONF_THRESHOLD:
                    self.recent_predictions.append(pred_label)
                else:
                    self.recent_predictions.append("neutral")

                # Stability Voting
                most_common, count = Counter(self.recent_predictions).most_common(1)[0]
                if count >= (self.STABILITY_FRAMES - 2):
                    self.stable_prediction = most_common

            except Exception as e:
                print(f"Inference Error: {e}")

        # 4. Form Analysis
        self.analyze_frame(self.stable_prediction, landmarks)
        
        return self.rep_counter, self.form_status, self.stable_prediction, self.debug_angles

    def analyze_frame(self, exercise_name, landmarks):
        if not landmarks: return
        
        # Convert to objects for easy dot-notation access in logic below
        class Point:
            def __init__(self, obj):
                self.x = obj['x'] if isinstance(obj, dict) else obj.x
                self.y = obj['y'] if isinstance(obj, dict) else obj.y
        
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

        try:
            # --- Rule-Based Logic (Extract Points) ---
            ls = lms_obj[11]; le = lms_obj[13]; lw = lms_obj[15]
            rs = lms_obj[12]; re = lms_obj[14]; rw = lms_obj[16]
            lh = lms_obj[23]; lk = lms_obj[25]; la = lms_obj[27]
            
            # Bicep Curl
            if exercise_name == 'bicepCurl':
                if lw.x < ls.x: # Simple visibility check (left side)
                    angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                else:
                    angle = calculate_angle_2d([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
                
                self.debug_angles = {'Elbow': int(angle)}
                
                if angle > 160: self.stage = "down"
                if angle < 45 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1

            # Squats
            elif 'Squat' in exercise_name:
                knee_angle = calculate_angle_2d([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])
                self.debug_angles = {'Knee': int(knee_angle)}
                
                if knee_angle > 160: self.stage = "up"
                if knee_angle < 100 and self.stage == 'up': self.stage = "down"
                if knee_angle > 160 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1

            # Lateral Raise
            elif exercise_name == 'lateralRaise':
                angle = calculate_angle_2d([lms_obj[24].x, lms_obj[24].y], [rs.x, rs.y], [re.x, re.y])
                self.debug_angles = {'Shoulder': int(angle)}
                if angle < 30: self.stage = "down"
                if angle > 80 and self.stage == 'down':
                    self.stage = "up"; self.rep_counter += 1

            # Presses
            elif 'Press' in exercise_name:
                elbow_angle = calculate_angle_2d([ls.x, ls.y], [le.x, le.y], [lw.x, lw.y])
                self.debug_angles = {'Elbow': int(elbow_angle)}
                if elbow_angle < 80: self.stage = "down"
                if elbow_angle > 150 and self.stage == "down":
                    self.stage = "up"; self.rep_counter += 1

        except Exception:
            pass

    def get_triggered_alert(self):
        alert = self.triggered_alert; self.triggered_alert = None; return alert

    def get_new_error_log(self):
        log = self.new_error_to_log; self.new_error_to_log = None; return log

    def reset_session(self):
        self.rep_counter = 0; self.stage = None; self.angle_sequence_buffer.clear()
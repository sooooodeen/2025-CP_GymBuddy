import pandas as pd
import numpy as np
import mediapipe as mp

# --- Helper Function to Calculate Angles (Corrected 3D Version) ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0
        
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- NEW: Pose Normalization Function ---
def normalize_pose(landmarks):
    """Normalizes landmarks to be invariant to position and scale."""
    # Convert landmark dictionaries to a NumPy array for easier calculations
    # We only need x, y, z for normalization
    coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])

    # 1. Find the center of the hips
    left_hip = coords[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = coords[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    hip_center = (left_hip + right_hip) / 2.0

    # 2. Calculate a scaling factor (torso size)
    left_shoulder = coords[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = coords[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    # Add a small epsilon to avoid division by zero
    torso_size = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6

    # 3. Center and scale the landmarks
    normalized_coords = (coords - hip_center) / torso_size

    # 4. Rebuild the landmark dictionary list with the normalized coordinates
    normalized_landmarks = []
    for i in range(len(landmarks)):
        normalized_landmarks.append({
            'x': normalized_coords[i, 0],
            'y': normalized_coords[i, 1],
            'z': normalized_coords[i, 2],
            'visibility': landmarks[i]['visibility']
        })
        
    return normalized_landmarks

# --- Load the Dataset ---
input_filename = 'exercise_coords_multi_angle.csv'
output_filename = 'exercise_coords_engineered.csv'

try:
    df = pd.read_csv(input_filename)
except FileNotFoundError:
    print(f"Error: '{input_filename}' not found. Please ensure your dataset is in the correct folder.")
    exit()

print("Dataset loaded. Starting feature engineering with pose normalization...")

# --- Feature Engineering Logic ---
engineered_data = []
mp_pose = mp.solutions.pose

for index, row in df.iterrows():
    new_row = row.to_dict()
    
    # 1. Extract landmarks from the row into a list of dictionaries
    landmarks_original = []
    for i in range(1, 34):
        landmarks_original.append({
            'x': row[f'x{i}'],
            'y': row[f'y{i}'],
            'z': row[f'z{i}'],
            'visibility': row[f'v{i}']
        })

    # 2. NORMALIZE THE POSE
    landmarks_normalized = normalize_pose(landmarks_original)

    # 3. Get key landmark coordinates for calculations FROM THE NORMALIZED DATA
    left_shoulder = [landmarks_normalized[mp_pose.PoseLandmark.LEFT_SHOULDER.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_SHOULDER.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_SHOULDER.value]['z']]
    left_elbow = [landmarks_normalized[mp_pose.PoseLandmark.LEFT_ELBOW.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_ELBOW.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_ELBOW.value]['z']]
    left_wrist = [landmarks_normalized[mp_pose.PoseLandmark.LEFT_WRIST.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_WRIST.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_WRIST.value]['z']]
    left_hip = [landmarks_normalized[mp_pose.PoseLandmark.LEFT_HIP.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_HIP.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_HIP.value]['z']]
    left_knee = [landmarks_normalized[mp_pose.PoseLandmark.LEFT_KNEE.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_KNEE.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_KNEE.value]['z']]
    left_ankle = [landmarks_normalized[mp_pose.PoseLandmark.LEFT_ANKLE.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_ANKLE.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.LEFT_ANKLE.value]['z']]

    right_shoulder = [landmarks_normalized[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]['z']]
    right_elbow = [landmarks_normalized[mp_pose.PoseLandmark.RIGHT_ELBOW.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_ELBOW.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_ELBOW.value]['z']]
    right_wrist = [landmarks_normalized[mp_pose.PoseLandmark.RIGHT_WRIST.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_WRIST.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_WRIST.value]['z']]
    right_hip = [landmarks_normalized[mp_pose.PoseLandmark.RIGHT_HIP.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_HIP.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_HIP.value]['z']]
    right_knee = [landmarks_normalized[mp_pose.PoseLandmark.RIGHT_KNEE.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_KNEE.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_KNEE.value]['z']]
    right_ankle = [landmarks_normalized[mp_pose.PoseLandmark.RIGHT_ANKLE.value]['x'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_ANKLE.value]['y'], landmarks_normalized[mp_pose.PoseLandmark.RIGHT_ANKLE.value]['z']]

    # 4. Calculate features using the normalized coordinates
    new_row['angle_left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    new_row['angle_left_shoulder'] = calculate_angle(left_hip, left_shoulder, left_elbow)
    new_row['angle_left_hip'] = calculate_angle(left_shoulder, left_hip, left_knee)
    new_row['angle_left_knee'] = calculate_angle(left_hip, left_knee, left_ankle)

    new_row['angle_right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
    new_row['angle_right_shoulder'] = calculate_angle(right_hip, right_shoulder, right_elbow)
    new_row['angle_right_hip'] = calculate_angle(right_shoulder, right_hip, right_knee)
    new_row['angle_right_knee'] = calculate_angle(right_hip, right_knee, right_ankle)

    new_row['dist_y_l_wrist_shoulder'] = abs(left_wrist[1] - left_shoulder[1])
    new_row['dist_y_r_wrist_shoulder'] = abs(right_wrist[1] - right_shoulder[1])

    new_row['dist_z_l_wrist_hip'] = abs(left_wrist[2] - left_hip[2])
    new_row['dist_z_r_wrist_hip'] = abs(right_wrist[2] - right_hip[2])
    
    engineered_data.append(new_row)

df_engineered = pd.DataFrame(engineered_data)
df_engineered.to_csv(output_filename, index=False)

print(f"✅ Feature engineering complete. New dataset with {len(df_engineered)} samples saved as '{output_filename}'")
import pandas as pd
import numpy as np
import mediapipe as mp

# --- Helper Function to Calculate Angles (Corrected 3D Version) ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points (in degrees)."""
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

# --- CORRECTED Pose Normalization Function ---
def normalize_pose_robust(landmarks, mp_pose_module):
    """Robust pose normalization using hip-to-shoulder torso length."""
    if len(landmarks) != 33:
        return np.full((33, 3), np.nan), np.nan, np.nan

    landmarks_np = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])

    if np.isnan(landmarks_np).any():
        return np.full((33, 3), np.nan), np.nan, np.nan

    left_hip = landmarks_np[mp_pose_module.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks_np[mp_pose_module.PoseLandmark.RIGHT_HIP.value]
    hip_center = (left_hip + right_hip) / 2.0

    left_shoulder = landmarks_np[mp_pose_module.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks_np[mp_pose_module.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    torso_length = np.linalg.norm(hip_center - shoulder_center) + 1e-6
    
    if torso_length < 1e-5:
        return np.full((33, 3), np.nan), np.nan, np.nan

    normalized_landmarks = (landmarks_np - hip_center) / torso_length
    
    return normalized_landmarks, hip_center, shoulder_center

# --- Main Script ---
mp_pose = mp.solutions.pose

# Define input filenames
cleaned_multi_angle_file = 'exercise_coords_multi_angle_cleaned.csv'
cleaned_anchor_file = 'exercise_coords_anchor_cleaned.csv'

# Define output filename for engineered features
output_filename = 'exercise_coords_engineered_v2.csv'

print("--- Starting Enhanced Feature Engineering Process ---")

# Load the primary cleaned dataset
try:
    multi_angle_df = pd.read_csv(cleaned_multi_angle_file)
    multi_angle_df['original_source_file'] = 'multi_angle_data' 
    print(f"Loaded '{cleaned_multi_angle_file}' with {len(multi_angle_df)} samples.")
except FileNotFoundError:
    print(f"Error: Cleaned Multi-Angle data '{cleaned_multi_angle_file}' not found. Please run data_cleaning.py first. Exiting.")
    exit()

# Load the optional cleaned anchor dataset
try:
    anchor_df = pd.read_csv(cleaned_anchor_file)
    anchor_df['original_source_file'] = 'anchor_data' 
    print(f"Loaded '{cleaned_anchor_file}' with {len(anchor_df)} samples.")
except FileNotFoundError:
    print(f"Warning: Cleaned Anchor data '{cleaned_anchor_file}' not found. Proceeding without anchor data.")
    anchor_df = pd.DataFrame()

# Combine the two dataframes if anchor data exists
if not anchor_df.empty:
    df = pd.concat([multi_angle_df, anchor_df], ignore_index=True)
    print(f"Combined cleaned dataset now has {len(df)} samples for feature engineering.")
else:
    df = multi_angle_df

# --- Standardize Class Names ---
df['class'] = df['class'].str.replace(r' - \d+', '', regex=True)
print("Standardized class names by removing numerical suffixes.")

print(f"Proceeding with {len(df)} cleaned samples.")
print("\nStarting feature engineering with robust pose normalization and expanded features...")

engineered_data = []

# No need for previous_angles_by_sequence when removing delta features
# No need for sorting as frame order doesn't matter for static features

for index, row in df.iterrows():
    new_row = {'class': row['class']}

    landmarks_original_list = []
    for i in range(33):
        landmarks_original_list.append({
            'x': row.get(f'x{i}', 0.0),
            'y': row.get(f'y{i}', 0.0),
            'z': row.get(f'z{i}', 0.0),
            'visibility': row.get(f'v{i}', 0.0),
        })

    normalized_coords_np, hip_center, shoulder_center = normalize_pose_robust(landmarks_original_list, mp_pose)

    if np.isnan(normalized_coords_np).any():
        continue 
    
    def get_norm_lm_coords(lm_index):
        return normalized_coords_np[lm_index]

    try:
        # Define Key Landmarks
        nose = get_norm_lm_coords(mp_pose.PoseLandmark.NOSE.value)
        left_ear = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_EAR.value)
        right_ear = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_EAR.value)
        
        left_shoulder = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_hip = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_HIP.value)
        
        left_elbow = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        left_wrist = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_WRIST.value)
        left_index = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_INDEX.value)
        right_elbow = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        right_wrist = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        right_index = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_INDEX.value)

        left_knee = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_KNEE.value)
        left_ankle = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        left_heel = get_norm_lm_coords(mp_pose.PoseLandmark.LEFT_HEEL.value)
        right_knee = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        right_ankle = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        right_heel = get_norm_lm_coords(mp_pose.PoseLandmark.RIGHT_HEEL.value)

        # --- Calculate Expanded Static Angles ---
        new_row['angle_torso_side_left'] = calculate_angle(left_shoulder, left_hip, left_knee)
        new_row['angle_torso_side_right'] = calculate_angle(right_shoulder, right_hip, right_knee)
        new_row['angle_torso_front'] = calculate_angle(left_shoulder, right_hip, right_shoulder)
        new_row['angle_neck'] = calculate_angle(nose, left_ear, right_ear)
        new_row['angle_spine_hip_shoulder_left'] = calculate_angle(left_hip, left_shoulder, right_shoulder)
        new_row['angle_spine_hip_shoulder_right'] = calculate_angle(right_hip, right_shoulder, left_shoulder)
        
        new_row['angle_left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
        new_row['angle_right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
        new_row['angle_left_shoulder_abduction'] = calculate_angle(left_hip, left_shoulder, left_elbow)
        new_row['angle_right_shoulder_abduction'] = calculate_angle(right_hip, right_shoulder, right_elbow)
        new_row['angle_left_wrist'] = calculate_angle(left_elbow, left_wrist, left_index)
        new_row['angle_right_wrist'] = calculate_angle(right_elbow, right_wrist, right_index)

        new_row['angle_left_hip'] = calculate_angle(left_shoulder, left_hip, left_knee)
        new_row['angle_right_hip'] = calculate_angle(right_shoulder, right_hip, right_knee)
        new_row['angle_left_knee'] = calculate_angle(left_hip, left_knee, left_ankle)
        new_row['angle_right_knee'] = calculate_angle(right_hip, right_knee, right_ankle)
        new_row['angle_left_ankle'] = calculate_angle(left_knee, left_ankle, left_heel)
        new_row['angle_right_ankle'] = calculate_angle(right_knee, right_ankle, right_heel)

        new_row['angle_shoulder_hip_twist_left'] = calculate_angle(right_shoulder, left_hip, right_hip)
        new_row['angle_shoulder_hip_twist_right'] = calculate_angle(left_shoulder, right_hip, left_hip)

        # --- Calculate Expanded Static Distances ---
        new_row['dist_shoulders'] = np.linalg.norm(left_shoulder - right_shoulder)
        new_row['dist_hips'] = np.linalg.norm(left_hip - right_hip)
        new_row['dist_left_wrist_knee'] = np.linalg.norm(left_wrist - left_knee)
        new_row['dist_right_wrist_knee'] = np.linalg.norm(right_wrist - right_knee)
        new_row['dist_left_elbow_hip'] = np.linalg.norm(left_elbow - left_hip)
        new_row['dist_right_elbow_hip'] = np.linalg.norm(right_elbow - right_hip)
        new_row['dist_left_ankle_wrist'] = np.linalg.norm(left_ankle - left_wrist)
        new_row['dist_right_ankle_wrist'] = np.linalg.norm(right_ankle - right_wrist)
        new_row['dist_nose_hip'] = np.linalg.norm(nose - hip_center)
        
        new_row['dist_y_l_wrist_shoulder'] = abs(left_wrist[1] - left_shoulder[1])
        new_row['dist_y_r_wrist_shoulder'] = abs(right_wrist[1] - right_shoulder[1])
        new_row['dist_y_l_hip_knee'] = abs(left_hip[1] - left_knee[1])
        new_row['dist_y_r_hip_knee'] = abs(right_hip[1] - right_knee[1])
        new_row['dist_y_l_shoulder_hip'] = abs(left_shoulder[1] - left_hip[1])
        new_row['dist_y_r_shoulder_hip'] = abs(right_shoulder[1] - right_hip[1])
        new_row['dist_y_l_ankle_heel'] = abs(left_ankle[1] - left_heel[1])
        new_row['dist_y_r_ankle_heel'] = abs(right_ankle[1] - right_heel[1])

        new_row['dist_z_l_wrist_hip'] = abs(left_wrist[2] - left_hip[2])
        new_row['dist_z_r_wrist_hip'] = abs(right_wrist[2] - right_hip[2])
        new_row['dist_z_l_shoulder_hip'] = abs(left_shoulder[2] - left_hip[2])
        new_row['dist_z_r_shoulder_hip'] = abs(right_shoulder[2] - right_hip[2])
        new_row['dist_z_nose_hip'] = abs(nose[2] - hip_center[2])

        if not any(np.isnan(list(new_row.values())[1:])):
            engineered_data.append(new_row)
            
    except IndexError as e:
        print(f"Skipping row {index} due to IndexError during landmark retrieval or angle calculation: {e}")
    except ValueError as e:
        print(f"Skipping row {index} due to ValueError during angle calculation (e.g., NaN input): {e}")
    except Exception as e:
        print(f"An unexpected error occurred at row {index}: {e}")

df_engineered = pd.DataFrame(engineered_data)
df_engineered = df_engineered.fillna(0)
df_engineered.to_csv(output_filename, index=False)

print(f"\n✅ Enhanced feature engineering complete. New dataset with {len(df_engineered)} samples saved as '{output_filename}'")
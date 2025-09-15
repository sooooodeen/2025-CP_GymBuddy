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

# --- Pose Normalization Function ---
def normalize_pose(landmarks):
    """Normalizes landmarks to be invariant to position and scale."""
    coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])

    # Check for NaN in coordinates that would break normalization
    if np.isnan(coords).any():
        return [{'x': np.nan, 'y': np.nan, 'z': np.nan, 'visibility': 0.0} for _ in landmarks]

    # Get hip and shoulder coordinates
    left_hip = coords[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = coords[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = coords[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = coords[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate normalization parameters
    hip_center = (left_hip + right_hip) / 2.0
    torso_size = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
    
    # If torso_size is still very small (e.g., if shoulders are very close or (0,0,0)),
    # normalization will produce NaNs or inf. Handle this gracefully.
    if torso_size < 1e-5: 
        return [{'x': np.nan, 'y': np.nan, 'z': np.nan, 'visibility': 0.0} for _ in landmarks]


    # Apply normalization
    normalized_coords = (coords - hip_center) / torso_size
    
    # Rebuild the landmark list with normalized coordinates
    normalized_landmarks = []
    for i in range(len(landmarks)):
        normalized_landmarks.append({
            'x': normalized_coords[i, 0],
            'y': normalized_coords[i, 1],
            'z': normalized_coords[i, 2],
            'visibility': landmarks[i]['visibility'],
        })

    return normalized_landmarks

# --- Main Script ---

# Define input filenames (NOW LOADING THE *CLEANED* FILES)
cleaned_multi_angle_file = 'exercise_coords_multi_angle_cleaned.csv'
cleaned_anchor_file = 'exercise_coords_anchor_cleaned.csv'

# Define output filename for engineered features
output_filename = 'exercise_coords_engineered.csv'

print("--- Starting Feature Engineering Process ---")

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
    anchor_df = pd.DataFrame() # Create an empty DataFrame if not found

# Combine the two dataframes if anchor data exists
if not anchor_df.empty:
    df = pd.concat([multi_angle_df, anchor_df], ignore_index=True)
    print(f"Combined cleaned dataset now has {len(df)} samples for feature engineering.")
else:
    df = multi_angle_df

# --- Standardize Class Names ---
# This will remove " - 1", " - 2", etc. from the class names
df['class'] = df['class'].str.replace(r' - \d+', '', regex=True)
print("Standardized class names by removing numerical suffixes.")

# --- REMOVED THE INLINE CLEANING LOGIC ---
# The cleaning is now handled by data_cleaning.py, which produces the files we just loaded.
print(f"Proceeding with {len(df)} cleaned samples.")

print("\nStarting feature engineering with pose normalization...")

engineered_data = []
mp_pose = mp.solutions.pose

for index, row in df.iterrows():
    new_row = {'class': row['class']}

    landmarks_original = []
    for i in range(1, 34): 
        # Since NaNs were filled with 0.0 by data_cleaning.py, 
        # we can safely use row.get(f'x{i}', 0.0) etc.
        # But it's good practice to ensure consistency with data_cleaning.py output.
        landmarks_original.append({
            'x': row.get(f'x{i}', 0.0),
            'y': row.get(f'y{i}', 0.0),
            'z': row.get(f'z{i}', 0.0),
            'visibility': row.get(f'v{i}', 0.0),
        })

    landmarks_normalized = normalize_pose(landmarks_original.copy())

    # This check is still necessary because normalize_pose can still produce NaNs
    # if critical landmarks become invalid during its calculations (e.g., torso_size=0).
    if np.isnan(landmarks_normalized[0]['x']): 
        continue 

    def get_norm_lm(lm_index):
        lm = landmarks_normalized[lm_index]
        return [lm['x'], lm['y'], lm['z']]

    try:
        left_shoulder = get_norm_lm(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        left_elbow = get_norm_lm(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        left_wrist = get_norm_lm(mp_pose.PoseLandmark.LEFT_WRIST.value)
        left_hip = get_norm_lm(mp_pose.PoseLandmark.LEFT_HIP.value)
        left_knee = get_norm_lm(mp_pose.PoseLandmark.LEFT_KNEE.value)
        left_ankle = get_norm_lm(mp_pose.PoseLandmark.LEFT_ANKLE.value)

        right_shoulder = get_norm_lm(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        right_elbow = get_norm_lm(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        right_wrist = get_norm_lm(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        right_hip = get_norm_lm(mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_knee = get_norm_lm(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        right_ankle = get_norm_lm(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        # --- Calculate Angles ---
        new_row['angle_left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
        new_row['angle_left_shoulder'] = calculate_angle(left_hip, left_shoulder, left_elbow)
        new_row['angle_left_hip'] = calculate_angle(left_shoulder, left_hip, left_knee)
        new_row['angle_left_knee'] = calculate_angle(left_hip, left_knee, left_ankle)

        new_row['angle_right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
        new_row['angle_right_shoulder'] = calculate_angle(right_hip, right_shoulder, right_elbow)
        new_row['angle_right_hip'] = calculate_angle(right_shoulder, right_hip, right_knee)
        new_row['angle_right_knee'] = calculate_angle(right_hip, right_knee, right_ankle)

        # --- Calculate Distances ---
        new_row['dist_y_l_wrist_shoulder'] = abs(left_wrist[1] - left_shoulder[1])
        new_row['dist_y_r_wrist_shoulder'] = abs(right_wrist[1] - right_shoulder[1])
        new_row['dist_z_l_wrist_hip'] = abs(left_wrist[2] - left_hip[2])
        new_row['dist_z_r_wrist_hip'] = abs(right_wrist[2] - right_hip[2])

        if not any(np.isnan(list(new_row.values())[1:])): 
            engineered_data.append(new_row)
    except IndexError as e:
        # This will be rare if data_cleaning.py did its job, but good to keep.
        print(f"Skipping row {index} due to IndexError during landmark retrieval or angle calculation: {e}")
    except ValueError as e:
        # This can still happen if calculate_angle receives invalid inputs,
        # but normalize_pose should have caught most NaN issues.
        print(f"Skipping row {index} due to ValueError during angle calculation (e.g., NaN input): {e}")


df_engineered = pd.DataFrame(engineered_data)
df_engineered.to_csv(output_filename, index=False)

print(f"\n✅ Feature engineering complete. New dataset with {len(df_engineered)} samples saved as '{output_filename}'")
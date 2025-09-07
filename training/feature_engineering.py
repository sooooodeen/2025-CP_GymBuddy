import pandas as pd
import numpy as np
import mediapipe as mp

# --- Helper Function to Calculate Angles ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points."""
    a = np.array(a) # First point
    b = np.array(b) # Mid point
    c = np.array(c) # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Load the Dataset ---
input_filename = 'exercise_coords_multi_angle.csv'
output_filename = 'exercise_coords_engineered.csv'

try:
    df = pd.read_csv(input_filename)
except FileNotFoundError:
    print(f"Error: '{input_filename}' not found. Please ensure your dataset is in the correct folder.")
    exit()

print("Dataset loaded. Starting feature engineering...")

# --- Feature Engineering Logic ---
# This list will hold the new, engineered data
engineered_data = []
mp_pose = mp.solutions.pose

for index, row in df.iterrows():
    # Create a dictionary to hold the new row data, starting with the original data
    new_row = row.to_dict()
    
    # Extract landmarks from the row
    landmarks = []
    for i in range(1, 34):
        landmark = {
            'x': row[f'x{i}'],
            'y': row[f'y{i}'],
            'z': row[f'z{i}'],
            'visibility': row[f'v{i}']
        }
        landmarks.append(landmark)

    # --- Calculate New Features (Angles and Distances) ---
    
    # Get key landmark coordinates for BOTH sides
    # We now use all 3 dimensions [x, y, z] for more robust calculations
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]['y'], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]['z']]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]['y'], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]['z']]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]['y'], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]['z']]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]['y'], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]['z']]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]['y'], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]['z']]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]['y'], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]['z']]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]['x'], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]['y'], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]['z']]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]['x'], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]['y'], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]['z']]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]['x'], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]['y'], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]['z']]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]['x'], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]['y'], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]['z']]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]['x'], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]['y'], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]['z']]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]['x'], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]['y'], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]['z']]

    # 1. Bilateral Body Angles (Now in 3D)
    new_row['angle_left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    new_row['angle_left_shoulder'] = calculate_angle(left_hip, left_shoulder, left_elbow)
    new_row['angle_left_hip'] = calculate_angle(left_shoulder, left_hip, left_knee)
    new_row['angle_left_knee'] = calculate_angle(left_hip, left_knee, left_ankle)

    new_row['angle_right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
    new_row['angle_right_shoulder'] = calculate_angle(right_hip, right_shoulder, right_elbow)
    new_row['angle_right_hip'] = calculate_angle(right_shoulder, right_hip, right_knee)
    new_row['angle_right_knee'] = calculate_angle(right_hip, right_knee, right_ankle)

    # 2. Key Vertical Distances (Y-axis)
    new_row['dist_y_l_wrist_shoulder'] = abs(left_wrist[1] - left_shoulder[1])
    new_row['dist_y_r_wrist_shoulder'] = abs(right_wrist[1] - right_shoulder[1])

    # 3. Key Horizontal Distances (Z-axis, for depth)
    new_row['dist_z_l_wrist_hip'] = abs(left_wrist[2] - left_hip[2])
    new_row['dist_z_r_wrist_hip'] = abs(right_wrist[2] - right_hip[2])
    
    engineered_data.append(new_row)

# Create a new DataFrame with the engineered features
df_engineered = pd.DataFrame(engineered_data)

# Save the new dataset
df_engineered.to_csv(output_filename, index=False)

print(f"✅ Feature engineering complete. New dataset with {len(df_engineered)} samples saved as '{output_filename}'")
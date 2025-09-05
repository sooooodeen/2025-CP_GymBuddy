import pandas as pd
import numpy as np
import mediapipe as mp

# --- Helper Function to Calculate Angles ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
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
    # Create a dictionary to hold the new row data
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
    
    # Get key landmark coordinates (we'll use the left side for consistency)
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]['y']]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]['y']]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]['y']]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]['y']]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]['y']]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]['x'], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]['y']]
    
    # 1. Key Body Angles
    new_row['angle_left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    new_row['angle_left_shoulder'] = calculate_angle(left_hip, left_shoulder, left_elbow)
    new_row['angle_left_hip'] = calculate_angle(left_shoulder, left_hip, left_knee)
    new_row['angle_left_knee'] = calculate_angle(left_hip, left_knee, left_ankle)
    
    # 2. Vertical Distances (to differentiate Upright Row from Curls)
    # This is a key feature for your confusion problem
    new_row['dist_y_wrist_shoulder'] = abs(left_wrist[1] - left_shoulder[1])
    new_row['dist_y_elbow_shoulder'] = abs(left_elbow[1] - left_shoulder[1])

    engineered_data.append(new_row)

# Create a new DataFrame with the engineered features
df_engineered = pd.DataFrame(engineered_data)

# Save the new dataset
df_engineered.to_csv(output_filename, index=False)

print(f"Feature engineering complete. New dataset saved as '{output_filename}'")

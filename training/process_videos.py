import os
import cv2
import mediapipe as mp
import pandas as pd
import uuid

# --- Define all 33 landmark names for the CSV header ---
LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# Create headers for X, Y, Z, and Visibility for each landmark
csv_header = ['class', 'sequence_id', 'frame_id', 'timestamp_ms']
for name in LANDMARK_NAMES:
    csv_header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_visibility'])

def process_videos_to_csv_with_landmarks(videos_directory, output_csv_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    data_rows = []
    
    print(f"Starting to process videos in: {videos_directory}")
    for root, dirs, files in os.walk(videos_directory):
        if not files:
            continue
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(video_path, videos_directory)
                path_parts = relative_path.split(os.path.sep)
                
                # Get class name from the folder name
                class_name = path_parts[0]
                
                print(f"Processing: {file}  =>  Assigned Label: '{class_name}'")
                sequence_id = uuid.uuid4()
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Could not open video file {video_path}")
                    continue
                    
                frame_id = 0
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        break
                    
                    # Get the timestamp for velocity calculation
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        # Extract all 33 landmarks
                        landmarks = results.pose_landmarks.landmark
                        
                        # Create a single row for the CSV
                        row = [class_name, sequence_id, frame_id, timestamp_ms]
                        for lm in landmarks:
                            row.extend([lm.x, lm.y, lm.z, lm.visibility])
                        
                        data_rows.append(row)
                        
                    frame_id += 1
                cap.release()
                
    if data_rows:
        print("\nCreating DataFrame and saving to CSV...")
        df = pd.DataFrame(data_rows, columns=csv_header)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved landmark data to {output_csv_path}")
    else:
        print("No data was extracted.")
    pose.close()

if __name__ == '__main__':
    videos_directory = 'raw_videos'
    output_csv_path = 'exercise_sequences_landmarks.csv' # New output file name
    process_videos_to_csv_with_landmarks(videos_directory, output_csv_path)
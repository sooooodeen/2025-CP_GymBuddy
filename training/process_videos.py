import os
import cv2
import mediapipe as mp
import pandas as pd
import uuid
import numpy as np # ADDED

# ADDED: 2D Normalization function
def normalize_landmarks_2d(landmarks):
    """Normalizes pose landmarks to 2D space based on their bounding box."""
    landmarks_np = np.array([[lm.x, lm.y] for lm in landmarks])

    min_coords = np.min(landmarks_np, axis=0)
    max_coords = np.max(landmarks_np, axis=0)
    
    center = (min_coords + max_coords) / 2.0
    scale = np.max(max_coords - min_coords)
    
    if scale < 1e-6:
        # Return a flat array of NaNs if scale is too small
        return np.full(66, np.nan) 
        
    normalized_landmarks = (landmarks_np - center) / scale
    # Return a flat array of 66 features (33 landmarks * 2 coords)
    return normalized_landmarks.flatten() 

def process_videos_to_csv(videos_directory, output_csv_path):
    """
    Processes all video files in a directory to extract MediaPipe pose landmarks
    and saves them to a single CSV file.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    data_rows = []
    
    # --- CHANGED: Create the header for the 2D CSV file (66 columns) ---
    header = ['class', 'sequence_id', 'frame_id']
    for i in range(33):
        header.extend([f'x_{i}', f'y_{i}'])

    print(f"Starting to process videos in: {videos_directory}")

    for root, dirs, files in os.walk(videos_directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                video_path = os.path.join(root, file)
                print(f"Processing: {video_path}")

                class_name = os.path.basename(video_path).split('_')[0]
                sequence_id = uuid.uuid4()
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Could not open video {video_path}")
                    continue

                frame_id = 0
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        break

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        # --- CHANGED: Normalize landmarks and save ---
                        normalized_landmarks = normalize_landmarks_2d(results.pose_landmarks.landmark)
                        
                        # Check if normalization was successful
                        if not np.isnan(normalized_landmarks).any():
                            row = [class_name, sequence_id, frame_id]
                            row.extend(normalized_landmarks)
                            data_rows.append(row)
                    
                    frame_id += 1
                cap.release()

    if data_rows:
        print("\nCreating DataFrame and saving to CSV...")
        df = pd.DataFrame(data_rows, columns=header)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved data to {output_csv_path}")
    else:
        print("No data was extracted. Please check your video files and directory.")
        
    pose.close()

# --- HOW TO USE ---
if __name__ == '__main__':
    videos_directory = 'raw_videos'
    output_csv_path = 'exercise_sequences.csv'
    process_videos_to_csv(videos_directory, output_csv_path)
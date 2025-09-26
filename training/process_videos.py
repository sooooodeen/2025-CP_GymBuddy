import os
import cv2
import mediapipe as mp
import pandas as pd
import uuid
import numpy as np

def calculate_angle(a, b, c):
    """Calculates the angle between three 3D landmarks."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_angle_features(landmarks):
    """Extracts a predefined set of biomechanical angles from pose landmarks."""
    lm = landmarks.landmark
    mp_pose = mp.solutions.pose.PoseLandmark

    # Define angles to calculate
    angles = {
        'left_elbow': calculate_angle(lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_ELBOW], lm[mp_pose.LEFT_WRIST]),
        'right_elbow': calculate_angle(lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_ELBOW], lm[mp_pose.RIGHT_WRIST]),
        'left_shoulder': calculate_angle(lm[mp_pose.LEFT_ELBOW], lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_HIP]),
        'right_shoulder': calculate_angle(lm[mp_pose.RIGHT_ELBOW], lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_HIP]),
        'left_hip': calculate_angle(lm[mp_pose.LEFT_SHOULDER], lm[mp_pose.LEFT_HIP], lm[mp_pose.LEFT_KNEE]),
        'right_hip': calculate_angle(lm[mp_pose.RIGHT_SHOULDER], lm[mp_pose.RIGHT_HIP], lm[mp_pose.RIGHT_KNEE]),
        'left_knee': calculate_angle(lm[mp_pose.LEFT_HIP], lm[mp_pose.LEFT_KNEE], lm[mp_pose.LEFT_ANKLE]),
        'right_knee': calculate_angle(lm[mp_pose.RIGHT_HIP], lm[mp_pose.RIGHT_KNEE], lm[mp_pose.RIGHT_ANKLE])
    }
    return angles

def process_videos_to_csv_with_angles(videos_directory, output_csv_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    data_rows = []
    
    # Define header based on the angle features
    feature_names = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
                     'left_hip', 'right_hip', 'left_knee', 'right_knee']
    header = ['class', 'sequence_id', 'frame_id'] + feature_names

    print(f"Starting to process videos in: {videos_directory}")

    for root, dirs, files in os.walk(videos_directory):
        if not files: continue

        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                video_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(video_path, videos_directory)
                path_parts = relative_path.split(os.path.sep)
                class_name_from_folder = path_parts[0]

                class_name = 'neutral' if class_name_from_folder.startswith('neutral') else class_name_from_folder

                print(f"Processing: {file}  =>  Assigned Label: '{class_name}'")
                sequence_id = uuid.uuid4()
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened(): continue

                frame_id = 0
                while cap.isOpened():
                    success, image = cap.read()
                    if not success: break

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        angles = extract_angle_features(results.pose_landmarks)
                        row = [class_name, sequence_id, frame_id] + list(angles.values())
                        data_rows.append(row)
                    
                    frame_id += 1
                cap.release()

    if data_rows:
        print("\nCreating DataFrame and saving to CSV...")
        df = pd.DataFrame(data_rows, columns=header)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved data to {output_csv_path}")
    else:
        print("No data was extracted.")
        
    pose.close()

if __name__ == '__main__':
    videos_directory = 'raw_videos'
    output_csv_path = 'exercise_sequences.csv'
    process_videos_to_csv_with_angles(videos_directory, output_csv_path)
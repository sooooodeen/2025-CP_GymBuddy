import os
import cv2
import mediapipe as mp
import pandas as pd
import uuid # To create a unique ID for each sequence (video)

def process_videos_to_csv(videos_directory, output_csv_path):
    """
    Processes all video files in a directory to extract MediaPipe pose landmarks
    and saves them to a single CSV file.

    Args:
        videos_directory (str): The path to the directory containing video files.
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Prepare list to hold all data rows
    data_rows = []
    
    # --- Create the header for the CSV file ---
    # First three columns: class, sequence_id, frame_id
    # Then, 33 landmarks with 3 coordinates (x, y, z) each = 99 columns
    header = ['class', 'sequence_id', 'frame_id']
    for i in range(33):
        header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    print(f"Starting to process videos in: {videos_directory}")

    # Walk through the directory to find all video files
    for root, dirs, files in os.walk(videos_directory):
        for file in files:
            # Check for common video file extensions
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                video_path = os.path.join(root, file)
                print(f"Processing: {video_path}")

                # --- Extract class name from filename ---
                # Assumes filename format: exerciseName_view_personID_takeNumber.mp4
                class_name = os.path.basename(video_path).split('_')[0]
                
                # Generate a unique ID for this video sequence
                sequence_id = uuid.uuid4()
                
                # --- Open the video file ---
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Could not open video {video_path}")
                    continue

                frame_id = 0
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        break # End of video

                    # Convert the BGR image to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Process the image and detect the pose
                    results = pose.process(image_rgb)
                    
                    # --- Extract and save landmarks if pose is detected ---
                    if results.pose_landmarks:
                        # Start the row with class, sequence ID, and frame ID
                        row = [class_name, sequence_id, frame_id]
                        
                        # Extract the x, y, z coordinates for all 33 landmarks
                        for landmark in results.pose_landmarks.landmark:
                            row.extend([landmark.x, landmark.y, landmark.z])
                        
                        # Add the completed row to our data list
                        data_rows.append(row)
                    
                    frame_id += 1

                cap.release()

    # --- Create a DataFrame and save to CSV ---
    if data_rows:
        print("\nCreating DataFrame and saving to CSV...")
        df = pd.DataFrame(data_rows, columns=header)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved data to {output_csv_path}")
    else:
        print("No data was extracted. Please check your video files and directory.")
        
    # Clean up MediaPipe resources
    pose.close()

# --- HOW TO USE ---
if __name__ == '__main__':
    # 1. DEFINE the path to your folder containing the raw videos.
    #    Organize your videos in subfolders if you like, the script will find them.
    videos_directory = 'raw_videos' # <-- IMPORTANT: CHANGE THIS

    # 2. DEFINE the name and path for your output CSV file.
    output_csv_path = 'exercise_sequences.csv' # <-- You can change this if you want

    # 3. RUN the script.
    process_videos_to_csv(videos_directory, output_csv_path)
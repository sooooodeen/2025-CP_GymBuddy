import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

DATASET_PATH = 'exercise_coords_engineered.csv'
WINDOW_NAME = 'Dataset Visualizer'
CANVAS_SIZE = (800, 800)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: Dataset file not found at '{DATASET_PATH}'")
    exit()

print(f"Dataset loaded with {len(df)} samples.")
print("\n--- Controls ---")
print("Press 'b' to go BACKWARD.")
print("Press any other key to go FORWARD.")
print("Press 'q' to QUIT.")

index = 0
while index < len(df):
    row = df.iloc[index]
    label = row['class']
    landmarks = []
    for i in range(1, 34):
        landmark = {
            'x': row[f'x{i}'],
            'y': row[f'y{i}'],
            'z': row[f'z{i}'],
            'visibility': row[f'v{i}']
        }
        landmarks.append(landmark)
    canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        pose_landmarks_proto.landmark.add(x=lm['x'], y=lm['y'], z=lm['z'], visibility=lm['visibility'])
    
    mp_drawing.draw_landmarks(
        image=canvas,
        landmark_list=pose_landmarks_proto,
        connections=mp_pose.POSE_CONNECTIONS
    )

    text_to_display = f"Index: {index} | Class: {label}"
    print(f"Displaying index: {index}, Class: {label}")
    cv2.putText(canvas, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(WINDOW_NAME, canvas)
    key = cv2.waitKey(0)
    
    if key == ord('q'):
        break
    elif key == ord('b'):
        index = max(0, index - 1)
    else:
        index += 1

cv2.destroyAllWindows()
print("Visualization finished.")
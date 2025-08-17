import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- SCRIPT SETUP ---
# Prompt the user to enter the angle they are recording
current_angle = input("Enter the angle you will be recording (e.g., 'front', 'side', '45_degree'): ")

if not current_angle:
    print("No angle provided. Exiting.")
    exit()

print(f"\n✅ Ready to record from the '{current_angle}' view.")

# Define the exercises and their corresponding keys
EXERCISE_KEY_MAP = {
    # Legs & Glutes
    's': 'squat',
    'r': 'romanian_deadlift',
    'l': 'lunge',
    't': 'step_up',
    'b': 'bulgarian_split_squat',
    'g': 'goblet_squat',
    # Back
    'e': 'bent_over_row',
    'd': 'deadlift',
    'o': 'one_arm_dumbbell_row',
    'w': 't_bar_row',
    'i': 'inverted_row',
    # Chest
    'c': 'bench_press',
    'f': 'dumbbell_chest_fly',
    'n': 'incline_press',
    'p': 'push_up',
    'x': 'decline_press',
    # Shoulders
    'h': 'overhead_press',
    'y': 'lateral_raise',
    'j': 'front_raise',
    'a': 'arnold_press',
    'u': 'upright_row',
    # Arms
    'z': 'bicep_curl',
    'm': 'hammer_curl',
    'k': 'tricep_kickback',
    'v': 'overhead_tricep_extension',
    'q': 'concentration_curl',
    # Bodyweight / Core
    '1': 'glute_bridge',
    '2': 'plank',
    '3': 'side_plank',
    '4': 'mountain_climber',
    '5': 'jump_squat'
}

# The name of our output CSV file
csv_file_name = 'exercise_coords_multi_angle.csv'

# Check if the CSV file exists. If not, create it and write the header.
if not os.path.exists(csv_file_name):
    header = ['class']
    for i in range(1, 34):
        header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
    
    with open(csv_file_name, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
# --- END SETUP ---

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

print("\n--- INSTRUCTIONS ---")
for key, exercise in EXERCISE_KEY_MAP.items():
    print(f"Press '{key}' to record {exercise}")
print("\nPress '0' to quit.") # Quit key is '0' to avoid conflicts

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Process the frame to find pose landmarks
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the skeleton
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Check for key presses
    key = cv2.waitKey(5) & 0xFF

    if key == ord('0'): # Changed quit key
        break
        
    # --- DATA EXPORT LOGIC ---
    if chr(key) in EXERCISE_KEY_MAP and results.pose_landmarks:
        # Get the base exercise name
        exercise_base_name = EXERCISE_KEY_MAP[chr(key)]
        # Create the new class label by combining exercise name and angle
        exercise_class_label = f"{exercise_base_name}_{current_angle}"
        
        print(f"Recording data for: {exercise_class_label}")
        
        # Extract landmarks and write to CSV
        landmarks = results.pose_landmarks.landmark
        row = [exercise_class_label]
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        with open(csv_file_name, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)
    # --- END DATA EXPORT ---

    cv2.imshow('AI Fitness Trainer - Data Collection', image)

cap.release()
cv2.destroyAllWindows()
print(f"\nData collection complete. Data saved to {csv_file_name}")
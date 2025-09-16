import pandas as pd

ENGINEERED_CSV_PATH = 'exercise_coords_engineered.csv'

FEATURE_COLUMNS = [
    'angle_left_elbow', 'angle_left_shoulder', 'angle_left_hip', 'angle_left_knee',
    'angle_right_elbow', 'angle_right_shoulder', 'angle_right_hip', 'angle_right_knee',
    'dist_y_l_wrist_shoulder', 'dist_y_r_wrist_shoulder',
    'dist_z_l_wrist_hip', 'dist_z_r_wrist_hip'
]
try:
    df = pd.read_csv(ENGINEERED_CSV_PATH)
    print(f"✅ Successfully loaded '{ENGINEERED_CSV_PATH}'")
except FileNotFoundError:
    print(f"❌ Error: Could not find the file '{ENGINEERED_CSV_PATH}'.")
    print("Please make sure you have run feature_engineering.py first.")
    exit()

while True:
    try:
        user_input = input("\nEnter the index number of the frame you want to check (or 'q' to quit): ")
        
        if user_input.lower() == 'q':
            break

        index_to_check = int(user_input)

        if not (0 <= index_to_check < len(df)):
            print(f"❌ Error: Index must be between 0 and {len(df) - 1}.")
            continue
        row = df.iloc[index_to_check]
        feature_vector = row[FEATURE_COLUMNS]
        
        print("\n" + "="*40)
        print(f"   FEATURES FOR INDEX: {index_to_check}")
        print(f"   EXERCISE CLASS:   {row['class']}")
        print("="*40)
        
        formatted_features = [f"{x:.2f}" for x in feature_vector.values]
        print(formatted_features)
        print("="*40 + "\n")

    except ValueError:
        print("❌ Error: Please enter a valid number.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

print("Exiting feature checker.")
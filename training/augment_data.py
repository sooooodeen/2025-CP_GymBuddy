import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_CSV = 'exercise_sequences_landmarks.csv'
OUTPUT_CSV = 'exercise_sequences_augmented.csv'

def augment_data_with_mirroring(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Identify all landmark columns (ending in _x, _y, _z, _visibility)
    all_cols = df.columns.tolist()
    landmark_cols = [c for c in all_cols if c.endswith(('_x', '_y', '_z', '_visibility'))]
    
    # 2. Find Left/Right pairs automatically
    # We look for 'left_' columns and try to find the matching 'right_' column
    swap_pairs = {}
    for col in landmark_cols:
        if 'left_' in col:
            right_col = col.replace('left_', 'right_')
            if right_col in all_cols:
                swap_pairs[col] = right_col
    
    print(f"identified {len(swap_pairs)} left/right landmark pairs to swap.")

    # 3. Create the Augmented Dataframe
    # We copy the original DF, generate new IDs, and apply transformations
    augmented_df = df.copy()
    
    # Update sequence IDs so they don't clash
    augmented_df['sequence_id'] = augmented_df['sequence_id'].apply(lambda x: str(x) + "_flip")
    
    print("Applying mirror transformation (Left/Right Swap + X-Flip)...")
    
    # Apply Swaps
    for left_col, right_col in swap_pairs.items():
        # Swap values
        temp = augmented_df[left_col].copy()
        augmented_df[left_col] = augmented_df[right_col]
        augmented_df[right_col] = temp

    # Apply X-Flip (Mirroring logic: x_new = 1 - x_old)
    # This simulates the camera being mirrored horizontally
    x_cols = [c for c in landmark_cols if c.endswith('_x')]
    augmented_df[x_cols] = 1.0 - augmented_df[x_cols]

    # 4. Combine and Save
    final_df = pd.concat([df, augmented_df], ignore_index=True)
    
    print(f"Original Rows: {len(df)}")
    print(f"New Total Rows: {len(final_df)}")
    
    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved augmented dataset to: {output_path}")

if __name__ == '__main__':
    augment_data_with_mirroring(INPUT_CSV, OUTPUT_CSV)
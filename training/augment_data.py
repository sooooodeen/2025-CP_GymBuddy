import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_CSV = 'exercise_sequences.csv'
OUTPUT_CSV = 'exercise_sequences_augmented.csv'
CLASSES_TO_AUGMENT = ['bentOverRow', 'bicepCurl', 'lateralRaise', 'shoulderPress', 'tricepKickback']

# NEW: Define the pairs of columns to swap for a horizontal flip
COLUMN_SWAP_PAIRS = {
    'left_elbow': 'right_elbow',
    'left_shoulder': 'right_shoulder',
    'left_hip': 'right_hip',
    'left_knee': 'right_knee'
}

def augment_data_with_angles(input_path, output_path):
    """
    Reads angle feature data from a CSV, creates horizontally flipped (mirrored)
    versions by swapping left/right columns, and saves a new augmented CSV.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    augmented_rows = []
    
    sequences_to_augment = df[df['class'].isin(CLASSES_TO_AUGMENT)]['sequence_id'].unique()
    
    print(f"Found {len(sequences_to_augment)} sequences to augment.")

    for seq_id in sequences_to_augment:
        sequence_df = df[df['sequence_id'] == seq_id].copy()
        
        augmented_seq_id = f"{seq_id}_aug_flip"
        sequence_df['sequence_id'] = augmented_seq_id
        
        # --- Perform the Augmentation by Swapping Columns ---
        for left_col, right_col in COLUMN_SWAP_PAIRS.items():
            # Store original left column in a temporary variable
            temp_left = sequence_df[left_col].copy()
            
            # Assign right column's data to the left column
            sequence_df[left_col] = sequence_df[right_col]
            
            # Assign original left column's data (from temp) to the right column
            sequence_df[right_col] = temp_left

        augmented_rows.append(sequence_df)

    if not augmented_rows:
        print("No rows were augmented. Saving original data.")
        df.to_csv(output_path, index=False)
        return

    print("Concatenating original and augmented data...")
    augmented_df = pd.concat(augmented_rows, ignore_index=True)
    final_df = pd.concat([df, augmented_df], ignore_index=True)

    print(f"Original number of sequences: {len(df['sequence_id'].unique())}")
    print(f"New total number of sequences: {len(final_df['sequence_id'].unique())}")
    
    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved augmented data to {output_path}")

if __name__ == '__main__':
    augment_data_with_angles(INPUT_CSV, OUTPUT_CSV)
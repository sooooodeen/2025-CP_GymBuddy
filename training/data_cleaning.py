import pandas as pd
import numpy as np
import os

print("--- Starting Data Cleaning Process ---")

# Define input filenames
multi_angle_file = 'exercise_coords_multi_angle.csv'
anchor_file = 'exercise_coords_anchor.csv'

# Define output filenames for the cleaned data
cleaned_multi_angle_file = 'exercise_coords_multi_angle_cleaned.csv'
cleaned_anchor_file = 'exercise_coords_anchor_cleaned.csv'

# --- Configuration for Cleaning ---
# Define critical landmark columns for filtering. These are the ones needed for pose normalization.
critical_landmark_coords = [
    'x12', 'y12', 'z12', # LEFT_SHOULDER (MediaPipe index 11)
    'x13', 'y13', 'z13', # RIGHT_SHOULDER (MediaPipe index 12)
    'x24', 'y24', 'z24', # LEFT_HIP (MediaPipe index 23)
    'x25', 'y25', 'z25', # RIGHT_HIP (MediaPipe index 24)
]

# Corresponding visibility columns for critical landmarks
critical_landmark_vis = [
    'v12', 'v13', 'v24', 'v25'
]

# Define a tolerance for float comparisons (e.g., values less than this are "near zero")
FUZZY_ZERO_TOLERANCE = 1e-4 # For values very close to 0.0
FUZZY_ONE_TOLERANCE = 1e-4  # For values very close to 1.0 (if they were placeholders)
MIN_VISIBILITY_THRESHOLD = 0.5 # Minimum visibility for a critical landmark to be considered valid

# --- Function to perform cleaning on a single DataFrame ---
def clean_dataframe(df, df_name):
    print(f"\nCleaning '{df_name}' with {len(df)} initial samples...")
    initial_rows = len(df)
    
    # Ensure critical columns exist
    existing_coords = [col for col in critical_landmark_coords if col in df.columns]
    existing_vis = [col for col in critical_landmark_vis if col in df.columns]

    if not existing_coords:
        print(f"Warning: Critical coordinate columns not found in '{df_name}'. Skipping coordinate-based cleaning.")
        return df # Return original if no critical columns

    df_processed = df.copy() # Work on a copy

    # Phase 1: Convert fuzzy placeholder 0s/1s in coordinates to NaN
    print(f"  Phase 1: Converting fuzzy 0s/1s to NaN for critical coordinates...")
    for i in range(len(critical_landmark_coords) // 3):
        x_col = critical_landmark_coords[i*3]
        y_col = critical_landmark_coords[i*3+1]
        z_col = critical_landmark_coords[i*3+2]
        
        if x_col in df_processed.columns and y_col in df_processed.columns and z_col in df_processed.columns:
            # Check if x,y,z are all near 0 or all near 1
            is_near_zeros = (df_processed[x_col].abs() < FUZZY_ZERO_TOLERANCE) & \
                            (df_processed[y_col].abs() < FUZZY_ZERO_TOLERANCE) & \
                            (df_processed[z_col].abs() < FUZZY_ZERO_TOLERANCE)
            
            is_near_ones = ((df_processed[x_col] - 1).abs() < FUZZY_ONE_TOLERANCE) & \
                           ((df_processed[y_col] - 1).abs() < FUZZY_ONE_TOLERANCE) & \
                           ((df_processed[z_col] - 1).abs() < FUZZY_ONE_TOLERANCE)
            
            # Replace these placeholder values with NaN
            df_processed.loc[is_near_zeros | is_near_ones, [x_col, y_col, z_col]] = np.nan
        else:
            print(f"    Warning: Missing some coordinate columns for landmark {i+1} in '{df_name}'.")

    # Phase 2: Convert coordinates to NaN if their corresponding critical visibility score is too low
    if existing_vis:
        print(f"  Phase 2: Applying visibility threshold (>{MIN_VISIBILITY_THRESHOLD}) to critical landmarks...")
        for vis_col in existing_vis:
            lm_index = int(vis_col[1:]) # Extract landmark number from 'vX'
            x_col = f'x{lm_index}'
            y_col = f'y{lm_index}'
            z_col = f'z{lm_index}'
            
            if vis_col in df_processed.columns:
                # If the visibility for this specific landmark is low, set its coords to NaN
                df_processed.loc[df_processed[vis_col] < MIN_VISIBILITY_THRESHOLD, [x_col, y_col, z_col]] = np.nan
            else:
                print(f"    Warning: Missing visibility column '{vis_col}' in '{df_name}'. Skipping for this column.")
    else:
        print(f"  Warning: Critical visibility columns not found in '{df_name}'. Skipping visibility-based cleaning.")

    # Phase 3: Drop rows where any critical landmark's coordinates are NaN (from original or our replacement)
    print(f"  Phase 3: Dropping rows with any NaN in critical coordinate columns...")
    df_cleaned = df_processed.dropna(subset=existing_coords).copy()
    
    # Fill any *remaining* NaNs (e.g., non-critical landmarks, or visibility scores) with 0.0
    # Use infer_objects(copy=False) to silence FutureWarning
    df_cleaned = df_cleaned.fillna(0.0).infer_objects(copy=False)

    rows_removed = initial_rows - len(df_cleaned)
    print(f"  Cleaned '{df_name}': Removed {rows_removed} rows. Remaining: {len(df_cleaned)} samples.")
    
    return df_cleaned


# --- Main execution ---

# Load and clean multi-angle data
try:
    multi_angle_df = pd.read_csv(multi_angle_file)
    cleaned_multi_angle_df = clean_dataframe(multi_angle_df, "Multi-Angle Data")
    cleaned_multi_angle_df.to_csv(cleaned_multi_angle_file, index=False)
    print(f"Cleaned Multi-Angle data saved to '{cleaned_multi_angle_file}'")
except FileNotFoundError:
    print(f"Error: '{multi_angle_file}' not found. Skipping multi-angle data cleaning.")
except Exception as e:
    print(f"An error occurred during multi-angle data cleaning: {e}")

# Load and clean anchor data
try:
    anchor_df = pd.read_csv(anchor_file)
    cleaned_anchor_df = clean_dataframe(anchor_df, "Anchor Data")
    cleaned_anchor_df.to_csv(cleaned_anchor_file, index=False)
    print(f"Cleaned Anchor data saved to '{cleaned_anchor_file}'")
except FileNotFoundError:
    print(f"Error: '{anchor_file}' not found. Skipping anchor data cleaning.")
except Exception as e:
    print(f"An error occurred during anchor data cleaning: {e}")

print("\n--- Data Cleaning Process Complete ---")
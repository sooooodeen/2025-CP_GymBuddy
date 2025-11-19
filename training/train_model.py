import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 90
RAW_DATA_CSV = 'exercise_sequences_landmarks.csv' 

# --- PATH FIX: Get the directory this script is in ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- FIX: Save files in the *current* script directory, not a new 'training' subfolder ---
LABEL_MAPPING_FILE = os.path.join(SCRIPT_DIR, 'label_mapping.json')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'exercise_classifier_bilstm.h5')
CONFUSION_MATRIX_FILE = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')
# --- END PATH FIX ---


# --- 1. FEATURE ENGINEERING ---

def calculate_angle(landmarks, p1_name, p2_name, p3_name):
    """Calculates the angle between three landmarks."""
    p1 = landmarks[[f'{p1_name}_x', f'{p1_name}_y', f'{p1_name}_z']].values
    p2 = landmarks[[f'{p2_name}_x', f'{p2_name}_y', f'{p2_name}_z']].values
    p3 = landmarks[[f'{p3_name}_x', f'{p3_name}_y', f'{p3_name}_z']].values
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate dot product and magnitudes
    dot = np.einsum('ij,ij->i', v1, v2) # Row-wise dot product
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)
    
    # Avoid division by zero
    mask = (mag1 != 0) & (mag2 != 0)
    cos_angle = np.zeros(len(dot))
    
    # Ensure cos_angle is not zero before division
    if cos_angle.size > 0:
        cos_angle[mask] = dot[mask] / (mag1[mask] * mag2[mask])
    
    # Clip values to avoid numerical instability
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def create_features(df):
    """Creates angle and velocity features from raw landmark data."""
    print("Calculating features...")
    features_df = pd.DataFrame()
    features_df['sequence_id'] = df['sequence_id']
    features_df['timestamp_ms'] = df['timestamp_ms']

    # --- 1. Define All Features ---
    angle_definitions = {
        'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
        'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
        'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip': ('right_shoulder', 'right_hip', 'right_knee'),
        
        'left_upper_arm': ('left_hip', 'left_shoulder', 'left_elbow'),
        'right_upper_arm': ('right_hip', 'right_shoulder', 'right_elbow'),
        
        # Torso (Upright vs. Bent-Over)
        'left_torso': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_torso': ('right_shoulder', 'right_hip', 'right_knee'),
        'torso_avg': ('left_shoulder', 'left_hip', 'left_knee'), # Placeholder
        
        # Knee (Squat vs. Hinge)
        'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
        'knee_avg': ('left_hip', 'left_knee', 'left_ankle'), # Placeholder
        
        # Hand Position (Upright Ambiguity)
        'left_wrist_shoulder_x_diff': ('nose', 'nose', 'nose'),
        'right_wrist_shoulder_x_diff': ('nose', 'nose', 'nose'),
        'left_wrist_elbow_y_diff': ('nose', 'nose', 'nose'),
        'right_wrist_elbow_y_diff': ('nose', 'nose', 'nose'),

        # --- NEW: Add Stance Width ---
        'stance_width': ('left_ankle', 'right_ankle', 'right_ankle') # Placeholder
    }
    
    angle_names = list(angle_definitions.keys())

    # --- 1A. Calculate all the ANGLE features first ---
    for name, (p1, p2, p3) in angle_definitions.items():
        if 'diff' in name: # Skip our new diff features for now
            continue
            
        if name == 'stance_width':
            # Calculate the simple X-axis distance between the ankles
            features_df[name] = np.abs(df['left_ankle_x'] - df['right_ankle_x'])
        
        elif name == 'torso_avg':
            left_torso = calculate_angle(df, 'left_shoulder', 'left_hip', 'left_knee')
            right_torso = calculate_angle(df, 'right_shoulder', 'right_hip', 'right_knee')
            features_df[name] = (left_torso + right_torso) / 2
        
        elif name == 'knee_avg':
            left_knee = calculate_angle(df, 'left_hip', 'left_knee', 'left_ankle')
            right_knee = calculate_angle(df, 'right_hip', 'right_knee', 'right_ankle')
            features_df[name] = (left_knee + right_knee) / 2
            
        else:
            features_df[name] = calculate_angle(df, p1, p2, p3)
            
    # --- 1B. Calculate new 'diff' features ---
    # This feature helps with lateralRaise vs. bicepCurl (sideways vs. forward)
    features_df['left_wrist_shoulder_x_diff'] = df['left_wrist_x'] - df['left_shoulder_x']
    features_df['right_wrist_shoulder_x_diff'] = df['right_wrist_x'] - df['right_shoulder_x']
    
    # This feature helps with uprightRow vs. shoulderPress (elbows high vs. low)
    features_df['left_wrist_elbow_y_diff'] = df['left_wrist_y'] - df['left_elbow_y']
    features_df['right_wrist_elbow_y_diff'] = df['right_wrist_y'] - df['right_elbow_y']

    # --- 1C. Clean up helper columns ---
    # We remove the individual left/right versions to only keep the 'avg'
    features_df = features_df.drop(columns=['left_torso', 'right_torso', 'left_knee', 'right_knee'])
    angle_names.remove('left_torso')
    angle_names.remove('right_torso')
    angle_names.remove('left_knee')
    angle_names.remove('right_knee')
    angle_names.remove('stance_width') # Don't calculate velocity for stance_width


    print("Calculating velocities...")
    # --- 2. Calculate Velocity ---
    # We group by sequence so the .diff() doesn't cross over different videos
    grouped = features_df.groupby('sequence_id')
    
    # Calculate time delta (dt) in seconds
    dt = grouped['timestamp_ms'].diff().fillna(0) / 1000.0
    
    # Calculate angular velocity (d_angle / dt)
    for name in angle_names:
        d_angle = grouped[name].diff().fillna(0)
        # Avoid division by zero on the first frame
        vel_col = f'{name}_vel'
        features_df[vel_col] = d_angle.div(dt).replace([np.inf, -np.inf], 0).fillna(0)

    # Get final list of all features
    feature_names = [col for col in features_df.columns if col not in ['sequence_id', 'timestamp_ms']]
    
    return features_df, feature_names

# --- 2. DATA AUGMENTATION ---

def jitter(sequence, sigma=0.03):
    """Adds small random noise to a sequence."""
    noise = np.random.normal(0, sigma, sequence.shape)
    return sequence + noise

def augment_data(X_list, y_list, num_augmentations=2):
    """Augments data with jittering."""
    X_augmented, y_augmented = [], []
    for seq, label in zip(X_list, y_list):
        # Add the original sequence
        X_augmented.append(seq)
        y_augmented.append(label)
        
        # Add augmented versions
        for _ in range(num_augmentations):
            jittered_seq = jitter(seq)
            X_augmented.append(jittered_seq)
            y_augmented.append(label)
            
    return X_augmented, y_augmented

# --- 3. MODEL TRAINING ---

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - Test Accuracy: {accuracy*100:.2f}%', fontsize=16)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    print(f"Confusion matrix saved to '{CONFUSION_MATRIX_FILE}'")

def main():
    print("Loading raw landmark data...")
    df = pd.read_csv(RAW_DATA_CSV)

    # Create features (angles + velocities)
    features_df, feature_names = create_features(df)
    
    # --- MEMORY FIX ---
    # Instead of merging, just copy the label and class from the original dataframe.
    features_df['class'] = df['class'] 
    # We already have sequence_id, so we just re-assign df to be features_df
    df = features_df 
    # --- END MEMORY FIX ---

    # Encode labels
    label_encoder = LabelEncoder()
    df['class_encoded'] = label_encoder.fit_transform(df['class'])
    
    label_mapping = {i: str(label) for i, label in enumerate(label_encoder.classes_)}
    with open(LABEL_MAPPING_FILE, 'w') as f:
        json.dump(label_mapping, f)
    print(f"Saved label mapping: {label_mapping}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    
    # --- FIX: Scale velocity, diff, and stance_width features, but NOT angles ---
    velocity_features = [f for f in feature_names if '_vel' in f]
    diff_features = [f for f in feature_names if '_diff' in f]
    
    # Add our new feature to the list
    features_to_scale = velocity_features + diff_features + ['stance_width']
    
    print(f"Scaling {len(features_to_scale)} features...")
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    # --- END FIX ---

    # --- DATA SPLITTING FIX: Split *before* augmentation ---
    
    # 1. Group data into sequences
    print("Grouping data into sequences...")
    original_sequences = []
    original_labels = []
    original_groups = [] # The unique sequence_id for each sequence
    
    for seq_id, group_df in df.groupby('sequence_id'):
        original_sequences.append(group_df[feature_names].values)
        original_labels.append(group_df['class_encoded'].iloc[0])
        original_groups.append(seq_id)

    # 2. Split based on sequence_id (groups)
    print("Splitting data into training and testing sets...")
    gs = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gs.split(original_sequences, original_labels, original_groups))
    
    X_train_orig = [original_sequences[i] for i in train_idx]
    y_train_orig = [original_labels[i] for i in train_idx]
    
    X_test_orig = [original_sequences[i] for i in test_idx]
    y_test_orig = [original_labels[i] for i in test_idx]

    # 3. Augment *only* the training data
    print(f"Original training sequences: {len(X_train_orig)}")
    X_train_aug, y_train_aug = augment_data(X_train_orig, y_train_orig, num_augmentations=2)
    print(f"Augmented training sequences: {len(X_train_aug)}")
    print(f"Test sequences: {len(X_test_orig)}")

    # 4. Pad *both* sets
    print("Padding sequences...")
    X_train = pad_sequences(X_train_aug, maxlen=SEQUENCE_LENGTH, padding='post', truncating='pre', dtype='float32')
    X_test = pad_sequences(X_test_orig, maxlen=SEQUENCE_LENGTH, padding='post', truncating='pre', dtype='float32')

    # 5. Categorize *both* sets
    num_classes = len(label_mapping)
    y_train = to_categorical(np.array(y_train_aug), num_classes=num_classes)
    y_test = to_categorical(np.array(y_test_orig), num_classes=num_classes)
    
    # --- END DATA SPLITTING FIX ---

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --- Build the NEW Bidirectional-LSTM model ---
    print("Building the Bi-LSTM model...")
    timesteps = X_train.shape[1]
    num_features = X_train.shape[2] 

    model = Sequential([
        tf.keras.layers.Input(shape=(timesteps, num_features)),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model.summary()

    # --- Train the model ---
    print("Training the model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

    history = model.fit(
        X_train, y_train,
        epochs=150, # Train for longer
        batch_size=32,
        validation_data=(X_test, y_test), # Use the actual test set for validation
        callbacks=[early_stopping, reduce_lr]
    )

    print("Evaluating the model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    plot_confusion_matrix(y_true, y_pred, classes=label_encoder.classes_)

    print(f"Saving the trained model to {MODEL_FILE}...")
    model.save(MODEL_FILE)
    print("Model saved.")

if __name__ == '__main__':
    main()
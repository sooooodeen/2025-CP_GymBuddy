import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Masking, Input, Attention, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 90
RAW_DATA_CSV = 'exercise_sequences_augmented.csv' 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_MAPPING_FILE = os.path.join(SCRIPT_DIR, 'label_mapping.json')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'exercise_classifier_bilstm.h5')
CONFUSION_MATRIX_FILE = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')
PADDING_VALUE = -10.0 

# --- 1. FEATURE ENGINEERING ---
def calculate_angle(landmarks, p1_name, p2_name, p3_name):
    """Calculates the angle between three landmarks."""
    p1 = landmarks[[f'{p1_name}_x', f'{p1_name}_y', f'{p1_name}_z']].values
    p2 = landmarks[[f'{p2_name}_x', f'{p2_name}_y', f'{p2_name}_z']].values
    p3 = landmarks[[f'{p3_name}_x', f'{p3_name}_y', f'{p3_name}_z']].values
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot = np.einsum('ij,ij->i', v1, v2)
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)
    
    mask = (mag1 != 0) & (mag2 != 0)
    cos_angle = np.zeros(len(dot))
    
    if cos_angle.size > 0:
        cos_angle[mask] = dot[mask] / (mag1[mask] * mag2[mask])
    
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle_deg

def create_features(df):
    print("Calculating features...")
    features_df = pd.DataFrame()
    features_df['sequence_id'] = df['sequence_id']
    features_df['timestamp_ms'] = df['timestamp_ms']

    # Define Angles
    angle_definitions = {
        'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
        'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
        'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip': ('right_shoulder', 'right_hip', 'right_knee'),
        'left_upper_arm': ('left_hip', 'left_shoulder', 'left_elbow'),
        'right_upper_arm': ('right_hip', 'right_shoulder', 'right_elbow'),
        'torso_avg': ('left_shoulder', 'left_hip', 'left_knee'), 
        'knee_avg': ('left_hip', 'left_knee', 'left_ankle'), 
    }
    
    # 1. Calculate Angles
    for name, (p1, p2, p3) in angle_definitions.items():
        if name == 'torso_avg':
            l = calculate_angle(df, 'left_shoulder', 'left_hip', 'left_knee')
            r = calculate_angle(df, 'right_shoulder', 'right_hip', 'right_knee')
            features_df[name] = (l + r) / 2
        elif name == 'knee_avg':
            l = calculate_angle(df, 'left_hip', 'left_knee', 'left_ankle')
            r = calculate_angle(df, 'right_hip', 'right_knee', 'right_ankle')
            features_df[name] = (l + r) / 2
        else:
            features_df[name] = calculate_angle(df, p1, p2, p3)

    # 2. Calculate Extra Features
    features_df['stance_width'] = np.abs(df['left_ankle_x'] - df['right_ankle_x'])
    features_df['left_wrist_shoulder_x_diff'] = df['left_wrist_x'] - df['left_shoulder_x']
    features_df['right_wrist_shoulder_x_diff'] = df['right_wrist_x'] - df['right_shoulder_x']
    features_df['left_wrist_elbow_y_diff'] = df['left_wrist_y'] - df['left_elbow_y']
    features_df['right_wrist_elbow_y_diff'] = df['right_wrist_y'] - df['right_elbow_y']

    # 3. Calculate Velocity (d_angle / dt)
    print("Calculating velocities...")
    angle_cols = list(angle_definitions.keys())
    grouped = features_df.groupby('sequence_id')
    dt = grouped['timestamp_ms'].diff().fillna(0) / 1000.0
    dt = dt.replace(0, 0.033) # Prevent divide by zero (assume 30fps)

    for name in angle_cols:
        d_angle = grouped[name].diff().fillna(0)
        features_df[f'{name}_vel'] = d_angle / dt

    # Collect valid feature names
    feature_names = [c for c in features_df.columns if c not in ['sequence_id', 'timestamp_ms']]
    return features_df, feature_names

# --- 2. AUGMENTATION ---
def jitter(sequence, sigma=0.03):
    return sequence + np.random.normal(0, sigma, sequence.shape)

def augment_data(X_list, y_list, num_augmentations=2):
    X_aug, y_aug = [], []
    for seq, label in zip(X_list, y_list):
        X_aug.append(seq)
        y_aug.append(label)
        for _ in range(num_augmentations):
            X_aug.append(jitter(seq))
            y_aug.append(label)
    return X_aug, y_aug

# --- 3. MAIN ---
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    print(f"Matrix saved to {CONFUSION_MATRIX_FILE}")

def main():
    print(f"Loading data from {RAW_DATA_CSV}...")
    if not os.path.exists(RAW_DATA_CSV):
        print("Data file not found. Please run augment_data.py first.")
        return

    df = pd.read_csv(RAW_DATA_CSV)
    
    # Create features
    features_df, feature_names = create_features(df)
    features_df['class'] = df['class']
    
    # Scale features
    print(f"Scaling {len(feature_names)} features...")
    scaler = StandardScaler()
    features_df[feature_names] = scaler.fit_transform(features_df[feature_names])

    # Encode Labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(features_df['class'])
    label_mapping = {i: str(c) for i, c in enumerate(le.classes_)}
    with open(LABEL_MAPPING_FILE, 'w') as f: json.dump(label_mapping, f)

    # Group into sequences
    sequences = []
    labels = []
    for _, group in features_df.groupby('sequence_id'):
        sequences.append(group[feature_names].values)
        labels.append(labels_encoded[group.index[0]])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Augment Train only
    print(f"Augmenting {len(X_train)} training sequences...")
    X_train, y_train = augment_data(X_train, y_train)

    # Pad
    X_train = pad_sequences(X_train, maxlen=SEQUENCE_LENGTH, padding='post', dtype='float32', value=PADDING_VALUE)
    X_test = pad_sequences(X_test, maxlen=SEQUENCE_LENGTH, padding='post', dtype='float32', value=PADDING_VALUE)
    
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Class Weights
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(enumerate(weights))

    # --- MODEL ---
    input_layer = Input(shape=(SEQUENCE_LENGTH, len(feature_names)))
    masked = Masking(mask_value=PADDING_VALUE)(input_layer)
    
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(masked)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Attention
    att = Attention()([x, x])
    pooled = GlobalAveragePooling1D()(att)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(pooled)
    output_layer = Dense(len(le.classes_), activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train
    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test_cat),
        class_weight=weight_dict,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
        ]
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == '__main__':
    main()
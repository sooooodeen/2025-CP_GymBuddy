import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Masking, Input, Attention, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score
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

# --- 1. FEATURE ENGINEERING (Enhanced) ---
def calculate_angle(landmarks, p1_name, p2_name, p3_name):
    p1 = landmarks[[f'{p1_name}_x', f'{p1_name}_y', f'{p1_name}_z']].values
    p2 = landmarks[[f'{p2_name}_x', f'{p2_name}_y', f'{p2_name}_z']].values
    p3 = landmarks[[f'{p3_name}_x', f'{p3_name}_y', f'{p3_name}_z']].values
    v1, v2 = p1 - p2, p3 - p2
    dot = np.einsum('ij,ij->i', v1, v2)
    mag1, mag2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    mask = (mag1 != 0) & (mag2 != 0)
    cos_angle = np.zeros(len(dot))
    if cos_angle.size > 0: cos_angle[mask] = dot[mask] / (mag1[mask] * mag2[mask])
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def create_features(df):
    print("Calculating features...")
    features_df = pd.DataFrame()
    features_df['sequence_id'] = df['sequence_id']
    features_df['timestamp_ms'] = df['timestamp_ms']

    # 1. ANGLES
    angle_definitions = {
        'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
        'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
        'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip': ('right_shoulder', 'right_hip', 'right_knee'),
        'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
        'torso_avg': ('left_shoulder', 'left_hip', 'left_knee'), 
    }
    
    for name, (p1, p2, p3) in angle_definitions.items():
        if name == 'torso_avg':
             l = calculate_angle(df, 'left_shoulder', 'left_hip', 'left_knee')
             r = calculate_angle(df, 'right_shoulder', 'right_hip', 'right_knee')
             features_df[name] = (l + r) / 2
        else:
             features_df[name] = calculate_angle(df, p1, p2, p3)
        
    # 2. STANCE RATIO
    ankle_dist = np.abs(df['left_ankle_x'] - df['right_ankle_x'])
    shoulder_dist = np.abs(df['left_shoulder_x'] - df['right_shoulder_x'])
    shoulder_dist = shoulder_dist.replace(0, 0.01)
    features_df['stance_ratio'] = ankle_dist / shoulder_dist

    # 3. RELATIVE COORDINATES (NEW!)
    # Calculate positions relative to the midpoint of the hips
    mid_hip_x = (df['left_hip_x'] + df['right_hip_x']) / 2
    mid_hip_y = (df['left_hip_y'] + df['right_hip_y']) / 2

    # Track key points relative to body center (normalization)
    key_points = ['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow', 'left_ankle', 'right_ankle', 'nose']
    for kp in key_points:
        features_df[f'{kp}_rel_x'] = df[f'{kp}_x'] - mid_hip_x
        features_df[f'{kp}_rel_y'] = df[f'{kp}_y'] - mid_hip_y

    # 4. VELOCITY
    # Calculate velocity for angles AND the new relative coordinates
    cols_to_track = list(angle_definitions.keys()) + [f'{kp}_rel_x' for kp in key_points] + [f'{kp}_rel_y' for kp in key_points]
    
    grouped = features_df.groupby('sequence_id')
    dt = grouped['timestamp_ms'].diff().fillna(0) / 1000.0
    dt = dt.replace(0, 0.033)
    
    for name in cols_to_track:
        d_val = grouped[name].diff().fillna(0)
        features_df[f'{name}_vel'] = d_val / dt

    feature_names = [col for col in features_df.columns if col not in ['sequence_id', 'timestamp_ms']]
    return features_df, feature_names

# --- 2. AUGMENTATION (Gentler) ---
def time_warp(sequence, rate_range=(0.9, 1.1)): 
    length, dims = sequence.shape
    rate = np.random.uniform(*rate_range)
    x_old = np.linspace(0, 1, length)
    x_new = np.linspace(0, 1, int(length * rate))
    new_sequence = np.zeros((len(x_new), dims))
    for i in range(dims):
        f = interp1d(x_old, sequence[:, i], kind='linear')
        new_sequence[:, i] = f(x_new)
    if len(new_sequence) > length:
        return new_sequence[:length]
    else:
        padding = np.tile(new_sequence[-1], (length - len(new_sequence), 1))
        return np.vstack([new_sequence, padding])

def jitter(sequence, sigma=0.03): 
    noise = np.random.normal(0, sigma, sequence.shape)
    return sequence + noise

def augment_data(X_list, y_list):
    X_augmented, y_augmented = [], []
    for seq, label in zip(X_list, y_list):
        X_augmented.append(seq)
        y_augmented.append(label)
        X_augmented.append(jitter(seq))
        y_augmented.append(label)
        X_augmented.append(time_warp(seq))
        y_augmented.append(label)
    return X_augmented, y_augmented

# --- 3. MAIN ---
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    print(f"Matrix saved to {CONFUSION_MATRIX_FILE}")

def main():
    print("Loading AUGMENTED data...")
    if not os.path.exists(RAW_DATA_CSV):
        print(f"Error: {RAW_DATA_CSV} not found. Run augment_data.py first!")
        return

    df = pd.read_csv(RAW_DATA_CSV)
    
    seq_counts = df['sequence_id'].value_counts()
    valid_seqs = seq_counts[seq_counts >= 15].index 
    df = df[df['sequence_id'].isin(valid_seqs)]
    
    features_df, feature_names = create_features(df)
    features_df['class'] = df['class']
    
    scaler = StandardScaler()
    features_df[feature_names] = scaler.fit_transform(features_df[feature_names])

    le = LabelEncoder()
    le.fit(features_df['class'])
    label_mapping = {i: str(c) for i, c in enumerate(le.classes_)}
    with open(LABEL_MAPPING_FILE, 'w') as f: json.dump(label_mapping, f)

    sequences, labels = [], []
    for seq_id, group in features_df.groupby('sequence_id'):
        sequences.append(group[feature_names].values)
        labels.append(le.transform([group['class'].iloc[0]])[0])

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.15, stratify=labels, random_state=42
    )

    print(f"Augmenting {len(X_train)} training sequences...")
    X_train, y_train = augment_data(X_train, y_train)

    X_train = pad_sequences(X_train, maxlen=SEQUENCE_LENGTH, padding='post', dtype='float32', value=PADDING_VALUE)
    X_test = pad_sequences(X_test, maxlen=SEQUENCE_LENGTH, padding='post', dtype='float32', value=PADDING_VALUE)
    
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(enumerate(weights))

    # --- MODEL ARCHITECTURE: Attention-Based Bi-LSTM ---
    input_layer = Input(shape=(SEQUENCE_LENGTH, len(feature_names)))
    
    masked_input = Masking(mask_value=PADDING_VALUE)(input_layer)
    
    # Slightly deeper network to handle new features
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)))(masked_input)
    lstm_out = Dropout(0.4)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)))(lstm_out)
    lstm_out = Dropout(0.4)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    
    # Attention Mechanism
    attention = Attention()([lstm_out, lstm_out])
    
    pooled = GlobalAveragePooling1D()(attention)
    
    dense = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(pooled)
    dense = Dropout(0.3)(dense)
    output_layer = Dense(len(le.classes_), activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training Final V2 Model (New Features)...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test_cat),
        class_weight=weight_dict,
        callbacks=[
            EarlyStopping(patience=25, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.00001)
        ]
    )

    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    plot_confusion_matrix(y_test, y_pred, le.classes_)
    
    model.save(MODEL_FILE)
    print("Model saved.")

if __name__ == '__main__':
    main()
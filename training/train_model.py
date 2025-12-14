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
import joblib 

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 90
PADDING_VALUE = -10.0 

# Paths setup (Uses the directory where this script is located)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_CSV = os.path.join(SCRIPT_DIR, 'exercise_sequences_augmented.csv') 
LABEL_MAPPING_FILE = os.path.join(SCRIPT_DIR, 'label_mapping.json')
MODEL_FILE_H5 = os.path.join(SCRIPT_DIR, 'exercise_classifier_bilstm.h5')
MODEL_FILE_TFLITE = os.path.join(SCRIPT_DIR, 'exercise_classifier_quant.tflite') # TFLite Output
SCALER_FILE = os.path.join(SCRIPT_DIR, 'scaler.pkl') 
CONFUSION_MATRIX_FILE = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')

# --- 1. ROBUST NORMALIZATION & FEATURE EXTRACTION ---

# Standard MediaPipe Body Landmark Mapping
MP_LANDMARKS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]

def get_landmark_array(row):
    """Converts a CSV row into a (33, 3) numpy array."""
    landmarks = []
    for name in MP_LANDMARKS:
        # Handle cases where Z might be missing in CSV by defaulting to 0
        x = row.get(f'{name}_x', 0.0)
        y = row.get(f'{name}_y', 0.0)
        z = row.get(f'{name}_z', 0.0)
        landmarks.append([x, y, z])
    return np.array(landmarks, dtype=np.float32)

def normalize_pose_robust(landmarks_np):
    """
    Matches the Web App Logic:
    Normalizes based on 2D Torso Length to avoid Z-axis noise.
    """
    # Indices: 23=L.Hip, 24=R.Hip, 11=L.Shoulder, 12=R.Shoulder
    try:
        left_hip = landmarks_np[23][:2]
        right_hip = landmarks_np[24][:2]
        hip_center_2d = (left_hip + right_hip) / 2.0

        left_shoulder = landmarks_np[11][:2]
        right_shoulder = landmarks_np[12][:2]
        shoulder_center_2d = (left_shoulder + right_shoulder) / 2.0

        # Scale based on X/Y only
        torso_length = np.linalg.norm(hip_center_2d - shoulder_center_2d) + 1e-6
        
        # Center based on 3D Hip
        hip_center_3d = (landmarks_np[23] + landmarks_np[24]) / 2.0
        normalized_landmarks = (landmarks_np - hip_center_3d) / torso_length
        return normalized_landmarks
    except:
        return np.zeros_like(landmarks_np)

def calculate_angle_3d(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    norm_ba = np.linalg.norm(ba); norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    dot_product = np.dot(ba, bc)
    cosine_angle = dot_product / (norm_ba * norm_bc)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def extract_features_exact_match(row):
    """
    Generates the EXACT 47 features used in analysis_logic.py
    """
    # 1. Get Raw
    raw_lms = get_landmark_array(row)
    
    # 2. Normalize (Robust 2D)
    norm_lms = normalize_pose_robust(raw_lms)
    
    def lm(i): return norm_lms[i]
    def dist(i, j): return np.linalg.norm(lm(i) - lm(j))
    hip_center = np.array([0.0, 0.0, 0.0])

    # --- 20 ANGLES ---
    angles = [
        calculate_angle_3d(lm(11), lm(23), lm(25)), calculate_angle_3d(lm(12), lm(24), lm(26)),
        calculate_angle_3d(lm(11), lm(24), lm(12)), calculate_angle_3d(lm(0), lm(7), lm(8)),
        calculate_angle_3d(lm(23), lm(11), lm(12)), calculate_angle_3d(lm(24), lm(12), lm(11)),
        calculate_angle_3d(lm(11), lm(13), lm(15)), calculate_angle_3d(lm(12), lm(14), lm(16)),
        calculate_angle_3d(lm(23), lm(11), lm(13)), calculate_angle_3d(lm(24), lm(12), lm(14)),
        calculate_angle_3d(lm(13), lm(15), lm(19)), calculate_angle_3d(lm(14), lm(16), lm(20)),
        calculate_angle_3d(lm(11), lm(23), lm(25)), calculate_angle_3d(lm(12), lm(24), lm(26)),
        calculate_angle_3d(lm(23), lm(25), lm(27)), calculate_angle_3d(lm(24), lm(26), lm(28)),
        calculate_angle_3d(lm(25), lm(27), lm(29)), calculate_angle_3d(lm(26), lm(28), lm(30)),
        calculate_angle_3d(lm(12), lm(23), lm(24)), calculate_angle_3d(lm(11), lm(24), lm(23))
    ]

    # --- 22 DISTANCES ---
    distances = [
        dist(11, 12), dist(23, 24), dist(15, 25), dist(16, 26),
        dist(13, 23), dist(14, 24), dist(27, 15), dist(28, 16),
        np.linalg.norm(lm(0) - hip_center),
        abs(lm(15)[1] - lm(11)[1]), abs(lm(16)[1] - lm(12)[1]),
        abs(lm(23)[1] - lm(25)[1]), abs(lm(24)[1] - lm(26)[1]),
        abs(lm(11)[1] - lm(23)[1]), abs(lm(12)[1] - lm(24)[1]),
        abs(lm(27)[1] - lm(29)[1]), abs(lm(28)[1] - lm(30)[1]),
        abs(lm(15)[2] - lm(23)[2]), abs(lm(16)[2] - lm(24)[2]),
        abs(lm(11)[2] - lm(23)[2]), abs(lm(12)[2] - lm(24)[2]),
        abs(lm(0)[2] - hip_center[2])
    ]
    
    # Combine (Total 42)
    feats = np.array(angles + distances, dtype=np.float32)
    
    # Pad to 47 (matches web app padding)
    feats = np.concatenate([feats, np.zeros(5, dtype=np.float32)])
    
    return feats

def create_dataset_from_csv(df):
    """Transforms raw CSV data into sequences of features."""
    print("Extracting features (This may take a moment)...")
    
    sequences = []
    labels = []
    
    # Process sequence by sequence
    for seq_id, group in df.groupby('sequence_id'):
        seq_features = []
        for _, row in group.iterrows():
            f = extract_features_exact_match(row)
            seq_features.append(f)
        
        if len(seq_features) > 0:
            sequences.append(np.array(seq_features))
            labels.append(group['class'].iloc[0])
            
    return sequences, labels

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
def main():
    print(f"Loading data from {RAW_DATA_CSV}...")
    if not os.path.exists(RAW_DATA_CSV):
        print(f"ERROR: Data file not found at {RAW_DATA_CSV}")
        return

    df = pd.read_csv(RAW_DATA_CSV)
    
    # 1. Extract Features aligned with Web App
    sequences, labels_raw = create_dataset_from_csv(df)
    
    if not sequences:
        print("ERROR: No sequences extracted. Check your CSV format.")
        return

    # 2. Flatten for Scaling
    # We need to scale all frames together
    all_frames = np.vstack(sequences)
    
    print(f"Fitting Scaler on {len(all_frames)} frames...")
    scaler = StandardScaler()
    all_frames_scaled = scaler.fit_transform(all_frames)
    
    # SAVE SCALER (CRITICAL STEP)
    joblib.dump(scaler, SCALER_FILE)
    print(f"✅ Scaler saved to {SCALER_FILE}")

    # Reshape back to sequences
    X = []
    curr_idx = 0
    for seq in sequences:
        seq_len = len(seq)
        X.append(all_frames_scaled[curr_idx : curr_idx + seq_len])
        curr_idx += seq_len
    
    # 3. Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    
    # Save Label Mapping
    label_mapping = {i: str(c) for i, c in enumerate(le.classes_)}
    with open(LABEL_MAPPING_FILE, 'w') as f: json.dump(label_mapping, f)
    print("✅ Label mapping saved.")

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Augment (Train only)
    print(f"Augmenting {len(X_train)} training sequences...")
    X_train, y_train = augment_data(X_train, y_train)

    # 6. Pad
    X_train = pad_sequences(X_train, maxlen=SEQUENCE_LENGTH, padding='post', dtype='float32', value=PADDING_VALUE)
    X_test = pad_sequences(X_test, maxlen=SEQUENCE_LENGTH, padding='post', dtype='float32', value=PADDING_VALUE)
    
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Class Weights
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(enumerate(weights))

    # --- MODEL ---
    input_shape = (SEQUENCE_LENGTH, 47) # 47 Features
    
    input_layer = Input(shape=input_shape)
    masked = Masking(mask_value=PADDING_VALUE)(input_layer)
    
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(masked)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
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
    
    model.save(MODEL_FILE_H5)
    print(f"✅ Keras Model saved to {MODEL_FILE_H5}")

    # --- CONVERT TO TFLITE ---
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Quantization
    tflite_model = converter.convert()

    with open(MODEL_FILE_TFLITE, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite Model saved to {MODEL_FILE_TFLITE}")
    # -------------------------

    # Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_MATRIX_FILE)
    print("✅ Confusion matrix saved.")

if __name__ == '__main__':
    main()
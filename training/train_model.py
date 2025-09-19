import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plots a confusion matrix using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

# --- 1. LOAD AND PREPROCESS THE DATA ---
print("Loading and preprocessing data...")

# Load the dataset
df = pd.read_csv('exercise_sequences.csv')

# Encode the class labels into numbers
label_encoder = LabelEncoder()
df['class_encoded'] = label_encoder.fit_transform(df['class'])

# Save the label mapping
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)
print(f"Saved label mapping: {label_mapping}")

# Isolate features (coordinates) and the target (class)
features = [col for col in df.columns if col.startswith(('x_', 'y_', 'z_'))]
X = df[features]
y = df['class_encoded']
sequence_ids = df['sequence_id']

# --- 2. GROUP DATA INTO SEQUENCES ---
# Group by sequence_id to form sequences of frames
print("Grouping data into sequences...")
sequences = []
labels = []
for seq_id in sequence_ids.unique():
    sequence_data = X[sequence_ids == seq_id]
    sequence_label = y[sequence_ids == seq_id].iloc[0]
    sequences.append(sequence_data.values)
    labels.append(sequence_label)

# --- 3. PAD SEQUENCES ---
# Pad sequences to ensure they all have the same length for the LSTM
print("Paddling sequences...")
# max_len = max(len(seq) for seq in sequences) # Optional: Can define a fixed max_len
# NEW VERSION
X_padded = pad_sequences(sequences, maxlen=90, padding='post', truncating='pre', dtype='float32')

# Convert labels to a numpy array and one-hot encode them
y_array = np.array(labels)
y_categorical = to_categorical(y_array, num_classes=len(label_mapping))

# --- 4. SPLIT THE DATA ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_categorical, test_size=0.2, random_state=42, stratify=y_array
)

print(f"Data shape: (Samples, Timesteps, Features)")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# --- 5. BUILD THE LSTM MODEL ---
print("Building the LSTM model...")
num_classes = len(label_mapping)
timesteps = X_train.shape[1]
num_features = X_train.shape[2]

model = Sequential([
    tf.keras.layers.Input(shape=(timesteps, num_features)),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 6. TRAIN THE MODEL ---
print("Training the model...")
# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50, # Increase epochs as needed
    batch_size=32,
    validation_split=0.2, # Use part of the training data for validation during training
    callbacks=[early_stopping]
)

# --- 7. EVALUATE THE MODEL ---
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Generate predictions for confusion matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Plot the confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=label_encoder.classes_)

# --- 8. SAVE THE MODEL ---
print("Saving the trained model...")
model.save('exercise_classifier_lstm.h5')
print("Model saved as exercise_classifier_lstm.h5")
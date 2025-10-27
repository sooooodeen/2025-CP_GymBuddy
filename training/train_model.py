import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

SEQUENCE_LENGTH = 90

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.1, f'Test Accuracy: {accuracy*100:.2f}%', ha='center', transform=plt.gca().transAxes)
    plt.show()

print("Loading and preprocessing data...")
df = pd.read_csv('exercise_sequences_augmented.csv')

label_encoder = LabelEncoder()
df['class_encoded'] = label_encoder.fit_transform(df['class'])

label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)
print(f"Saved label mapping: {label_mapping}")

features = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
]
X = df[features]
y = df['class_encoded']
sequence_ids = df['sequence_id']

print("Grouping data into sequences...")
sequences = []
labels = []
for seq_id in sequence_ids.unique():
    sequence_data = X[sequence_ids == seq_id]
    sequence_label = y[sequence_ids == seq_id].iloc[0]
    sequences.append(sequence_data.values)
    labels.append(sequence_label)

print("Padding sequences...")
X_padded = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH, padding='post', truncating='pre', dtype='float32')

y_array = np.array(labels)
y_categorical = to_categorical(y_array, num_classes=len(label_mapping))

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_categorical, test_size=0.2, random_state=42, stratify=y_array
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

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

print("Training the model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)

print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

plot_confusion_matrix(y_true, y_pred, classes=label_encoder.classes_)

print("Saving the trained model...")
model.save('exercise_classifier_lstm.h5')
print("Model saved.")
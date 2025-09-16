import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the ENGINEERED Dataset ---
dataset_filename = 'exercise_coords_engineered.csv'

try:
    df = pd.read_csv(dataset_filename)
except FileNotFoundError:
    print(f"Error: '{dataset_filename}' not found.")
    print("Please make sure you have run the feature_engineering.py script first.")
    exit()

print(f"Engineered dataset loaded successfully with {len(df)} samples.")

# --- 2. Prepare the data ---
# We explicitly define our feature set to be ONLY the engineered columns.
feature_cols = [
    'angle_left_elbow', 'angle_left_shoulder', 'angle_left_hip', 'angle_left_knee',
    'angle_right_elbow', 'angle_right_shoulder', 'angle_right_hip', 'angle_right_knee',
    'dist_y_l_wrist_shoulder', 'dist_y_r_wrist_shoulder',
    'dist_z_l_wrist_hip', 'dist_z_r_wrist_hip'
]

X = df[feature_cols]
y = df['class']

# --- NEW: Save feature names ---
# This is crucial for ensuring the live prediction script uses the exact same feature order.
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print(f"✅ Feature names saved to 'feature_names.pkl'")
# --- END NEW ---

# Split data into training and testing sets
# stratify=y ensures the class distribution is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- NEW STEP: Scale the features for the Neural Network ---
print("\nScaling features for Neural Network training...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# IMPORTANT: Save the scaler so you can load it in AI_pose_corrector.py
# The live webcam data MUST be scaled with this exact same scaler before prediction.
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Feature scaler saved successfully as 'scaler.pkl'")


# --- 3. Build and Train the Neural Network (MLPClassifier) Model ---
print("\nTraining a Neural Network (MLPClassifier) model...")

mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50), # Two hidden layers: 100 neurons in first, 50 in second. Experiment with this!
    activation='relu',             # ReLU is a common and effective activation function
    solver='adam',                 # Adam is an efficient optimization algorithm
    alpha=0.0001,                  # L2 regularization parameter to prevent overfitting
    batch_size='auto',             # Automatically determines batch size
    learning_rate_init=0.001,      # Initial learning rate
    max_iter=500,                  # Maximum number of training iterations (epochs). May need more or less.
    random_state=42,
    verbose=True                   # Print training progress (loss per epoch)
)

# Fit the model using the SCALED training data
mlp_model.fit(X_train_scaled, y_train)
model = mlp_model # Assign to 'model' for consistency with the rest of the script
print("Model training complete.")


# --- 4. Evaluate the Model ---
print("\nEvaluating the new MLP model...")
# Make predictions using the SCALED test data
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ MLP Model Accuracy on Engineered Data: {accuracy * 100:.2f}%")


# --- 5. Save the Trained Model ---
# Renaming the model file to indicate it's an MLP
model_filename = 'exercise_model_mlp.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"\n✅ New MLP model saved successfully as '{model_filename}'")


# --- 6. Visualize a Confusion Matrix ---
print("\nGenerating confusion matrix visualization...")
try:
    # Ensure labels match model.classes_ order
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix for MLP Model') # Updated title
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_mlp.png') # Updated filename
    print("✅ Confusion matrix saved as 'confusion_matrix_mlp.png'")
except Exception as e:
    print(f"Could not generate confusion matrix plot. Error: {e}")
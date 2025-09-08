import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 3. Build and Train the Model ---
print("\nTraining a SIMPLER, more robust model...")
# By setting max_depth and min_samples_leaf, we prevent the model from overfitting.
# It can't build overly complex "rules" and is forced to generalize.
model = RandomForestClassifier(
    n_estimators=100,       # Keep the number of trees
    max_depth=10,           # IMPORTANT: Limit the depth of each tree to 10 levels
    min_samples_leaf=5,     # IMPORTANT: Each final "leaf" must have at least 5 samples
    random_state=42
)
model.fit(X_train, y_train)
print("Model training complete.")


# --- 4. Evaluate the Model ---
print("\nEvaluating the new model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy on Engineered Data: {accuracy * 100:.2f}%")


# --- 5. Save the Trained Model ---
model_filename = 'exercise_model_engineered.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"\n✅ New, smarter model saved successfully as '{model_filename}'")


# --- 6. Visualize a Confusion Matrix ---
print("\nGenerating confusion matrix visualization...")
try:
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix for Engineered Model')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_engineered.png')
    print("✅ Confusion matrix saved as 'confusion_matrix_engineered.png'")
except Exception as e:
    print(f"Could not generate confusion matrix plot. Error: {e}")
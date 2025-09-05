import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the ENGINEERED Dataset ---
# This now points to your new, smarter dataset file
dataset_filename = 'exercise_coords_engineered.csv'

try:
    df = pd.read_csv(dataset_filename)
except FileNotFoundError:
    print(f"Error: '{dataset_filename}' not found.")
    print("Please make sure you have run the feature_engineering.py script first.")
    exit()

print(f"Engineered dataset loaded successfully with {len(df)} samples.")

# --- 2. Prepare the data ---
# The features (X) now include all the original coordinates PLUS the new engineered angles and distances
X = df.drop('class', axis=1) 
y = df['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 3. Build and Train the Model ---
print("\nTraining the model on the engineered dataset...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")


# --- 4. Evaluate the Model ---
print("\nEvaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy on Engineered Data: {accuracy * 100:.2f}%")


# --- 5. Save the Trained Model ---
# We save it with a new name to distinguish it from the old model
model_filename = 'exercise_model_engineered.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"\n✅ New, smarter model saved successfully as '{model_filename}'")


# --- 6. (Optional) Visualize a Confusion Matrix ---
print("\nGenerating confusion matrix visualization...")
try:
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix for Engineered Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_engineered.png')
    print("✅ Confusion matrix saved as 'confusion_matrix_engineered.png'")
except Exception as e:
    print(f"Could not generate confusion matrix plot. Error: {e}")


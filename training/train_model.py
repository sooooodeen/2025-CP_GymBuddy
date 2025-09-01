import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the Dataset ---
dataset_filename = 'exercise_coords_multi_angle.csv'

try:
    df = pd.read_csv(dataset_filename)
except FileNotFoundError:
    print(f"Error: '{dataset_filename}' not found.")
    print("Please make sure your dataset file is in the same folder as this script.")
    exit()

print(f"Dataset loaded successfully with {len(df)} samples.")

df = df[~df['class'].str.contains('rigth side', na=False)]


# --- 2. Prepare the Data ---
X = df.drop('class', axis=1) 
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 3. Build and Train the Model ---
print("\nTraining the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")


# --- 4. Evaluate the Model ---
print("\nEvaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")


# --- 5. Save the Trained Model ---
model_filename = 'exercise_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"\n✅ Model saved successfully as '{model_filename}'")


# --- 6. (Optional) Visualize a Confusion Matrix ---
print("\nGenerating confusion matrix visualization...")
try:
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("✅ Confusion matrix saved as 'confusion_matrix.png'")
except Exception as e:
    print(f"Could not generate confusion matrix plot. Error: {e}")
    print("This may happen if you don't have a graphical backend. The model is still saved.")

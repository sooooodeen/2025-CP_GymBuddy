import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# The name of the dataset file you just provided
dataset_filename = 'exercise_coords_multi_angle.csv'

# 1. Load the dataset
try:
    df = pd.read_csv(dataset_filename)
except FileNotFoundError:
    print(f"Error: '{dataset_filename}' not found.")
    print("Please make sure your dataset file is in the same folder as this script.")
    exit()

# Optional: Clean up the one outlier class with a typo if it exists
df = df[df['class'] != 'dumbbell_incline_press_rigth side']


# 2. Prepare the data
# Features (all columns except 'class')
X = df.drop('class', axis=1) 
# Labels (the 'class' column)
y = df['class']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 3. Build and Train the Model
print("Training the model on your dataset...")
# We use a RandomForestClassifier, a strong choice for this type of data.
model = RandomForestClassifier(n_estimators=100, random_state=42)
# The .fit() method is where the model learns the patterns from the data.
model.fit(X_train, y_train)
print("Model training complete.")


# 4. Evaluate the Model
print("\nEvaluating the model...")
# Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# 5. Save the Trained Model
# We use pickle to save the model object to a file for later use.
model_filename = 'exercise_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel saved successfully as '{model_filename}'")
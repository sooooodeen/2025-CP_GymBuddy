import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- Load the Trained Model ---
model_filename = 'exercise_model.pkl'
try:
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_filename}'")
    print("Please make sure the model file is in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Start Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("--- Live feed started. Press 'q' to quit. ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Process the frame for pose detection
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- Prediction Logic ---
    if results.pose_landmarks:
        # 1. Extract and format keypoints
        landmarks = results.pose_landmarks.landmark
        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # Make a prediction
        X = np.array([row]) # Convert to numpy array in the correct shape
        predicted_class = model.predict(X)[0]
        confidence = model.predict_proba(X)[0]
        
        # 2. Display the results on the frame
        # Get the class with the highest probability
        max_confidence_index = np.argmax(confidence)
        predicted_class_name = model.classes_[max_confidence_index]
        prediction_confidence = confidence[max_confidence_index]
        
        # Create a status box
        cv2.rectangle(image, (0, 0), (350, 60), (245, 117, 16), -1)
        
        # Display the predicted class
        cv2.putText(image, 'CLASS', (15, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, predicted_class_name, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the prediction confidence
        cv2.putText(image, 'CONFIDENCE', (220, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'{prediction_confidence:.2f}', (215, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw the skeleton
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Display the final image
    cv2.imshow('AI Fitness Trainer', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
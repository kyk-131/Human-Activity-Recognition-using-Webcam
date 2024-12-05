import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from collections import Counter

# Set dataset path
DATASET_PATH = r"C:\Users\latha\Action_Recognition\data\WIS\video"  # Change this path to your dataset location

# Parameters
n_frames = 16  # Number of frames to use per video
frame_size = (64, 64)  # Resize frames to 64x64

# Function to load and preprocess HMDB-51 dataset
def load_hmdb51_data(dataset_path):
    video_data = []
    labels = []
    classes = os.listdir(dataset_path)
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip if not a directory

        for video_file in os.listdir(class_path):
            if video_file.endswith('.avi'):  # Filter HMDB-51 video files
                video_path = os.path.join(class_path, video_file)
                frames = extract_frames(video_path)
                
                # Ensure enough frames are available
                if len(frames) >= n_frames:
                    video_data.append(frames[:n_frames])  # Take first 'n_frames'
                    labels.append(class_name)
                else:
                    print(f"Skipping {video_file}: Not enough frames.")
    
    print(f"Loaded {len(video_data)} videos.")  # Check how many videos are loaded
    print(f"Classes found: {set(labels)}")  # Check unique classes
    return np.array(video_data), np.array(labels)

# Function to extract frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Load data
print("Loading HMDB-51 dataset...")
X, y = load_hmdb51_data(DATASET_PATH)

# Check the first few labels to confirm loading
print(f"Labels: {y[:10]}")  # Print the first 10 labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

# Check the unique classes in the labels
print(f"Unique Classes: {len(set(y))}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Normalize and reshape data
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, n_frames, 64, 64, 1)
X_test = X_test.reshape(-1, n_frames, 64, 64, 1)

# Build a simple CNN-LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(n_frames, 64, 64, 1)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train, y_train, batch_size=8, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Real-time recognition (using webcam)
cap = cv2.VideoCapture(0)
recognized_actions = []

frame_buffer = []  # Buffer to store frames for prediction

def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame / 255.0  # Normalize
    return frame.reshape(64, 64, 1)

print("Starting real-time recognition...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame_processed = preprocess_frame(frame)

    # Add the processed frame to the buffer
    frame_buffer.append(frame_processed)

    # If the buffer has 16 frames, make a prediction
    if len(frame_buffer) == 16:
        # Convert the buffer to the correct shape (1, 16, 64, 64, 1)
        input_data = np.array(frame_buffer).reshape(1, 16, 64, 64, 1)
        
        # Predict the action
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        action_label = label_encoder.inverse_transform([predicted_class])[0]

        recognized_actions.append(action_label)

        # Display the recognized action on the frame
        cv2.putText(frame, action_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Clear the buffer after making the prediction
        frame_buffer = []

    # Display the frame with the action label
    cv2.imshow("Real-Time Action Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()

# Plot recognized actions during webcam session
print("Plotting recognized actions...")
from collections import Counter
action_counts = Counter(recognized_actions)
plt.figure(figsize=(10, 6))
plt.bar(action_counts.keys(), action_counts.values(), color='skyblue')
plt.title("Actions Recognized During Webcam Session")
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

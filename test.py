import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt

# Set the dataset path
DATASET_PATH = r"C:\Users\latha\Human-Activity-Recognition-using-Webcam\data\UCF-101"
n_steps = 32  # Sequence length

# Load and preprocess the UCF101 dataset
def load_ucf101_data(dataset_path):
    """
    Load video data and corresponding labels from the UCF101 dataset.

    Args:
        dataset_path (str): Path to the UCF101 dataset.

    Returns:
        np.ndarray: Processed video data.
        np.ndarray: Corresponding labels.
    """
    video_data = []
    labels = []

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    # Iterate through class directories
    classes = os.listdir(dataset_path)
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip non-directory files

        for video_file in os.listdir(class_path):
            if not video_file.endswith(('.avi', '.mp4')):  # Filter video files
                continue

            video_path = os.path.join(class_path, video_file)
            try:
                # Extract frames from the video
                frames = extract_frames(video_path)
                if len(frames) >= n_steps:
                    video_data.append(frames[:n_steps])  # Trim to n_steps
                    labels.append(class_name)
                else:
                    print(f"Skipping {video_file}: Not enough frames.")
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    return np.array(video_data), np.array(labels)


# Extract frames from a video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))  # Resize frames
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Load data
print("Loading dataset...")
X, y = load_ucf101_data(DATASET_PATH)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = tf.keras.utils.to_categorical(y_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Normalize and reshape data
X_train = X_train / 255.0  # Normalize
X_test = X_test / 255.0
X_train = X_train.reshape(-1, n_steps, 64, 64, 1)
X_test = X_test.reshape(-1, n_steps, 64, 64, 1)

# Define the LSTM-CNN model
model = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(n_steps, 64, 64, 1)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 4
num_epochs = 5

print("Training the model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_test, y_test))

# Evaluate the model
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Real-time action recognition
cap = cv2.VideoCapture(0)  # Webcam input
recognized_actions = []

def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame / 255.0  # Normalize
    return frame.reshape(1, 1, 64, 64, 1)  # Shape for the model

print("Starting real-time recognition...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_frame(frame)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    action_label = label_encoder.inverse_transform([predicted_class])[0]

    recognized_actions.append(action_label)

    cv2.putText(frame, action_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Action Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()

# Visualize recognized actions
action_counts = Counter(recognized_actions)
plt.figure(figsize=(10, 6))
plt.bar(action_counts.keys(), action_counts.values(), color='skyblue')
plt.title("Actions Recognized During Webcam Session")
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

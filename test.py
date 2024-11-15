import numpy as np
import tensorflow as tf
import time
import cv2
import matplotlib.pyplot as plt

# Useful Constants
LABELS = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"
]

DATASET_PATH = "data/HAR_pose_activities/database/"
X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"
y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"

n_steps = 32  # 32 timesteps per series

# Load the networks inputs
def load_X(X_path):
    with open(X_path, 'r') as file:
        X_ = np.array([elem for elem in [row.split(',') for row in file]], dtype=np.float32)
    blocks = int(len(X_) / n_steps)
    X_ = np.array(np.split(X_, blocks))
    return X_

# Load the networks outputs
def load_y(y_path):
    with open(y_path, 'r') as file:
        y_ = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)
    return y_ - 1  # for 0-based indexing

X_train = load_X(X_train_path)
X_test = load_X(X_test_path)
y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# Ensure labels are of correct shape
y_train = np.squeeze(y_train)  # Remove unnecessary dimensions
y_test = np.squeeze(y_test)    # Remove unnecessary dimensions

# Input Data
training_data_count = len(X_train)
test_data_count = len(X_test)
n_input = len(X_train[0][0])  # num input parameters per timestep
n_hidden = 34  # Hidden layer num of features
n_classes = 6

# Learning rate settings
init_learning_rate = 0.005
decay_rate = 0.96
decay_steps = 100000

lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset, i.e., 300 epochs
batch_size = 512

# One hot encoding function
def one_hot(y_):
    return tf.one_hot(y_, depth=n_classes)

# Dataset preparation
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, one_hot(y_train)))
train_dataset = train_dataset.batch(batch_size).repeat()  # Add repeat() to ensure infinite dataset

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, one_hot(y_test)))
test_dataset = test_dataset.batch(batch_size)

# Define Custom LSTM Layer using tf.keras.layers
class CustomLSTMLayer(tf.keras.layers.Layer):
    def __init__(self, n_hidden, n_classes):
        super(CustomLSTMLayer, self).__init__()
        self.n_hidden = n_hidden
        self.out_dense = tf.keras.layers.Dense(n_classes)

        # Initialize the LSTM cells only once, not in the call method
        self.lstm_cells = [tf.keras.layers.LSTMCell(self.n_hidden) for _ in range(2)]  # 2 LSTM layers
        self.lstm_layer = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(self.lstm_cells), return_sequences=True)

    def call(self, inputs):
        lstm_output = self.lstm_layer(inputs)
        pooled_output = tf.keras.layers.GlobalAveragePooling1D()(lstm_output)
        return self.out_dense(pooled_output)

# Define input tensor shape for model
X_input = tf.keras.Input(shape=(n_steps, n_input), dtype=tf.float32, name="X_input")
y_input = tf.keras.Input(shape=(n_classes,), dtype=tf.float32, name="y_input")

# Apply the LSTM network
lstm_layer = CustomLSTMLayer(n_hidden=n_hidden, n_classes=n_classes)
pred = lstm_layer(X_input)  # Output predictions from the LSTM network

# Define the model using the functional API
model = tf.keras.Model(inputs=X_input, outputs=pred)

# Learning rate settings and optimizer
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(init_learning_rate, decay_steps, decay_rate, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Compile model with optimizer and loss function
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Time tracking for training
time_start = time.time()

# Train the model
model.fit(train_dataset, epochs=30, steps_per_epoch=training_data_count // batch_size)

# After training, evaluate on the test set
final_loss, final_acc = model.evaluate(test_dataset, batch_size=batch_size)
print(f"FINAL TEST RESULTS: Loss = {final_loss:.6f}, Accuracy = {final_acc:.6f}")

# Real-time video action recognition
cap = cv2.VideoCapture(0)  # Start webcam

# Preprocessing function for the webcam frame
def preprocess_frame(frame):
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize frame to match the input dimensions (32x36 in this case)
    frame_resized = cv2.resize(frame_gray, (36, 32))
    # Ensure correct dimensions
    frame_resized = np.expand_dims(frame_resized, axis=-1)  # Add channel dimension
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    return frame_resized / 255.0  # Normalize if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for action recognition
    input_data = preprocess_frame(frame)

    # Predict the action
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    # Display the predicted action on the frame
    label = LABELS[predicted_class]
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with prediction
    cv2.imshow('Action Recognition', frame)

    # Exit the video window on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
X_test = X_test.reshape(-1, n_steps, n_input)

# Perform prediction
predictions = model.predict(X_test)

# Process prediction output
predictions_classes = np.argmax(predictions, axis=1)

from sklearn import metrics
print("Testing Accuracy: {}%".format(100 * final_acc))
print("")
print("Precision: {}%".format(100 * metrics.precision_score(y_test, predictions_classes, average="weighted")))
print("Recall: {}%".format(100 * metrics.recall_score(y_test, predictions_classes, average="weighted")))
print("f1_score: {}%".format(100 * metrics.f1_score(y_test, predictions_classes, average="weighted")))

# Confusion Matrix
print("")
print("Confusion Matrix:")
print("Created using test set of {} datapoints, normalised to % of each class in the test dataset".format(len(y_test)))
confusion_matrix = metrics.confusion_matrix(y_test, predictions_classes)

# Normalizing the confusion matrix
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

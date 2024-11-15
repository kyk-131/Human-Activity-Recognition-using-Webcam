README: Human Activity Recognition using LSTM
Overview
This project implements a Human Activity Recognition (HAR) system using a Long Short-Term Memory (LSTM) network with real-time webcam input. The model recognizes six activities: Jumping, Jumping Jacks, Boxing, Waving (2 hands), Waving (1 hand), and Clapping Hands. The dataset is based on motion sensor data captured from human activities, and the system is designed to classify these actions using deep learning.

The key components of the system include:

Dataset Preprocessing: Loading, reshaping, and normalizing data.
LSTM Model Architecture: A custom LSTM layer stacked for action classification.
Real-time Video Action Recognition: Webcam input for live predictions.
Evaluation: The model is evaluated using standard metrics such as accuracy, precision, recall, and F1 score.
Requirements
Python 3.x
TensorFlow 2.x
NumPy
OpenCV
Matplotlib
scikit-learn
To install the required libraries, run:

bash
Copy code
pip install tensorflow numpy opencv-python matplotlib scikit-learn
Dataset
The dataset used is HAR Pose Activities with time-series data. The dataset includes:

X_train.txt and X_test.txt: Feature data containing time-series for each activity.
Y_train.txt and Y_test.txt: Ground truth labels for the activities.
Dataset Structure
Each sample in the dataset is divided into 32 timesteps, with each timestep containing multiple features. The target labels are integer-encoded, and the system converts them into a one-hot encoded format for classification.

Code Description
1. Loading Data
The data is loaded from text files and reshaped into a format compatible with the LSTM model. The following functions are responsible for loading and preprocessing the input data:

load_X(X_path): Loads feature data.
load_y(y_path): Loads label data.
2. Custom LSTM Layer
A custom LSTM layer is defined using TensorFlow’s tf.keras.layers.Layer. The LSTM consists of two stacked LSTM cells, followed by a dense layer for final classification output.

python
Copy code
class CustomLSTMLayer(tf.keras.layers.Layer):
    def __init__(self, n_hidden, n_classes):
        super(CustomLSTMLayer, self).__init__()
        self.n_hidden = n_hidden
        self.out_dense = tf.keras.layers.Dense(n_classes)
        self.lstm_cells = [tf.keras.layers.LSTMCell(self.n_hidden) for _ in range(2)]  # 2 LSTM layers
        self.lstm_layer = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(self.lstm_cells), return_sequences=True)
3. Model Training
The LSTM network is trained on the dataset using TensorFlow's model.fit(). The learning rate is decayed exponentially during training. The model is compiled with the categorical_crossentropy loss function and Adam optimizer.

4. Real-time Video Action Recognition
Using OpenCV, the webcam is initialized to capture real-time video frames. The frames are preprocessed by resizing, converting to grayscale, and normalizing before being passed to the trained model for action recognition.

python
Copy code
cap = cv2.VideoCapture(0)
The predictions are displayed live on the video frames with labels indicating the recognized action.

5. Model Evaluation
The model is evaluated on the test dataset using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
The results are plotted as a confusion matrix to visualize the performance of the model.

python
Copy code
from sklearn import metrics
print("Precision: {}%".format(100 * metrics.precision_score(y_test, predictions_classes, average="weighted")))
Steps to Run
Prepare the Dataset: Download and place the dataset (X_train.txt, X_test.txt, Y_train.txt, Y_test.txt) in the data/HAR_pose_activities/database/ directory.
Run the Script: Execute the script to train the model and start the real-time action recognition.
bash
Copy code
python har_action_recognition.py
Exit: Press 'q' to exit the real-time webcam window.
Example Output
The model will print the test results, including accuracy, precision, recall, and F1-score, followed by a confusion matrix and a visualization of it.

Sample Output:
vbnet
Copy code
FINAL TEST RESULTS: Loss = 0.132145, Accuracy = 0.931234
Testing Accuracy: 93.12%
Precision: 93.45%
Recall: 92.78%
f1_score: 93.11%
Confusion Matrix:
Created using test set of 1000 datapoints, normalised to % of each class in the test dataset
Notes
The system assumes that the webcam is available and connected.
The model's performance is evaluated on the test set after training.
This implementation uses TensorFlow 2.x and OpenCV for real-time video processing.

### Overview

This project implements an **action recognition system** using a combination of **Convolutional Neural Networks (CNNs)** and **Long Short-Term Memory networks (LSTMs)**. It processes video datasets to predict actions based on temporal and spatial features.

#### Key Features:
- **Data Processing**: The system extracts frames from videos, resizes them, and prepares them for training and prediction.
- **Model Architecture**: A CNN extracts spatial features from frames, while an LSTM captures the temporal dependencies between frames.
- **Real-Time Recognition**: Recognize actions live using a webcam or by uploading video files.
- **Visualization**: Displays recognized actions and their frequencies through a bar chart for better interpretability.

Workflow:
1. **Training**:
   - The dataset is loaded, preprocessed, and split into training and testing sets.
   - A CNN-LSTM model is trained to classify actions based on video frames.

2. **Prediction**:
   - For real-time action recognition, the webcam captures frames and predicts actions in intervals.
   - Uploaded videos are analyzed, and predictions are displayed as a bar chart.

3. **User Interaction**:
   - A simple **Tkinter-based GUI** allows users to:
     - Train the model on their dataset.
     - Perform live action recognition using a webcam.
     - Upload videos for action recognition.

This project provides a modular framework that can be adapted to other action recognition tasks by changing the dataset or tweaking the model architecture.

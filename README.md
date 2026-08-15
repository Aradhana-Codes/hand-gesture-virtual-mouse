# Hand Gesture Based Virtual Mouse Controller with Personalization
A machine learning-based virtual mouse controller that allows users to interact with a computer using hand gestures captured through a webcam.
The system uses MediaPipe to detect 21 hand landmarks, extracts normalized landmark coordinates, and uses a Random Forest classifier to recognize hand gestures. Recognized gestures are then mapped to mouse actions using PyAutoGUI.
The project also includes a customizable data collection process, allowing users to add their own hand gesture samples to the existing dataset and retrain the model for improved recognition of their gesture patterns.

## Project Objective

The main objective of this project is to create a touch-free computer interaction system that allows users to control basic mouse operations using hand gestures.
The project combines: Computer Vision + Feature Engineering + Machine Learning + Human-Computer Interaction
        
## Features
- Real-time hand detection using MediaPipe
- Extraction of 21 hand landmarks
- 63 numerical features from hand landmarks (21 × X, Y, Z coordinates)
- Random Forest-based gesture classification
- Confidence-based gesture prediction
- Mouse control using PyAutoGUI
- Real-time webcam visualization
- Custom gesture data collection
- Ability to add user-specific samples to the existing dataset
- Model and label encoder saved using Joblib

## How the Project Works
<p align="center">
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/bfb33669-7e9e-4ce7-9f15-eab6a035221c" />
</p>

## File Description
### 1. collect_data.py
This script collects hand gesture data using the webcam.
The user enters the gesture name: pinch, peace, fist and palm
MediaPipe detects the hand and extracts its 21 landmarks.
For each landmark, the X, Y, and Z coordinates are calculated relative to the wrist landmark.
This produces: 21 landmarks × 3 coordinates = 63 features
The gesture label is then added to the end of each row.

The collected samples are appended to: dataset/Hand Gesture Landmark Coordinates Dataset.csv
Samples are collected every 0.2 seconds, resulting in approximately 5 samples per second while a hand is detected.
Press q to stop data collection.

#### Landmark Feature Extraction
MediaPipe provides 21 hand landmarks.
Each landmark contains: X coordinate, Y coordinate and Z coordinate
Instead of using absolute coordinates, the project normalizes them relative to the wrist:
landmarks.append(lm.x - wrist.x)
landmarks.append(lm.y - wrist.y)
landmarks.append(lm.z - wrist.z)
This makes the features less dependent on the absolute position of the hand in the camera frame.
The resulting feature vector contains: 63 numerical features + 1 gesture label

### 2. train_base_model.py

This script trains the machine learning model using the collected dataset.
Training Process
CSV Dataset
     ↓
Load Dataset
     ↓
Separate Features and Labels
     ↓
Label Encoding
     ↓
Train-Test Split
     ↓
Random Forest Training
     ↓
Accuracy Evaluation
     ↓
Save Model
The dataset is divided into:
80% → Training data
20% → Testing data
using: train_test_split(X,y_encoded,test_size=0.2,random_state=42)

#### Random Forest Classifier
The project uses: RandomForestClassifier(n_estimators=300)
The Random Forest model learns the relationship between the 63 hand landmark features and the corresponding gesture classes.
The model calculates:
Training accuracy
Test accuracy
and saves the trained model to: models/base_model.pkl The label encoder is saved to: models/label_encoder.pkl using Joblib.

#### 3. virtual_mouse.py

This is the main real-time application.
It loads: models/base_model.pkl, models/label_encoder.pkl and starts the webcam.
For every detected hand:
MediaPipe detects the hand.
21 landmarks are extracted.
Landmark coordinates are normalized relative to the wrist.
The 63 features are passed to the Random Forest model.
The model predicts the gesture.
Prediction confidence is calculated.
If confidence is above the threshold, the corresponding mouse action is performed.

The current confidence threshold is: confidence_threshold = 0.85

This helps prevent low-confidence predictions from triggering mouse actions.

#### Personalization / Custom Data Collection
One of the features of this project is its customizable gesture recognition.
The project allows a user to collect additional samples of their own hand gestures.

For example:
Existing Dataset
       +
User's New Gesture Samples
       ↓
Updated CSV Dataset
       ↓
Retrain Random Forest
       ↓
Updated Gesture Recognition Model

This means the model can learn from the user's own hand landmark patterns.

Important: The current implementation does not automatically adapt the model during every use.
Instead, personalization works through:
Collect personal samples, Add samples to dataset, Retrain model and Use updated model

Therefore, the project is best described as a customizable/personalized gesture recognition system rather than a continuously self-learning system.

## Technologies Used
Technology	              Purpose
Python	                  Main programming language
OpenCV	                  Webcam capture and image processing
MediaPipe	          Hand detection and landmark extraction
NumPy	                  Numerical feature processing
Pandas	                  Dataset loading and processing
Scikit-learn	          Random Forest machine learning model
PyAutoGUI	          Mouse and scroll control
Joblib	                  Saving and loading trained models
CSV	                  Gesture dataset storage

## Installation
### 1. Clone the Repository
git clone https://github.com/Aradhana-Codes/hand-gesture-virtual-mouse.git

Navigate to the project directory:
cd hand-gesture-virtual-mouse

### 2. Create a Virtual Environment
Windows: python -m venv venv
Activate it: venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

## How to Run

### Step 1 — Collect Gesture Data
Run: python collect_data.py
Enter one of the supported gesture names:
pinch
peace
fist
palm

Example: Enter gesture name (pinch, peace, fist, palm): palm
Perform the gesture in front of the webcam.
The samples will automatically be saved to: dataset/Hand Gesture Landmark Coordinates Dataset.csv

Press: q to stop collecting.
Repeat the process for the other gestures.

### Step 2 — Train the Model
After collecting or adding gesture samples, run:
python train_base_model.py
The script will:

Load the CSV dataset
Encode gesture labels
Split the dataset
Train the Random Forest classifier
Calculate training accuracy
Calculate test accuracy
Save the trained model

The generated files are:
models/base_model.pkl
models/label_encoder.pkl

### Step 3 — Start the Virtual Mouse
Run: python virtual_mouse.py
The webcam will open and the system will begin recognizing gestures.
Press: q to exit.


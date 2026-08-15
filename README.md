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
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/bfb33669-7e9e-4ce7-9f15-eab6a035221c" />


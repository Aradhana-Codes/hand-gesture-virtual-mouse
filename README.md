# Hand Gesture Based Virtual Mouse Controller with Personalization
A machine learning-based virtual mouse controller that allows users to interact with a computer using hand gestures captured through a webcam.
The system uses MediaPipe to detect 21 hand landmarks, extracts normalized landmark coordinates, and uses a Random Forest classifier to recognize hand gestures. Recognized gestures are then mapped to mouse actions using PyAutoGUI.
The project also includes a customizable data collection process, allowing users to add their own hand gesture samples to the existing dataset and retrain the model for improved recognition of their gesture patterns.

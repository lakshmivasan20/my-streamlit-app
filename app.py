import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Streamlit App
st.title("Real-Time Dance Pose Detection")
st.write("Using OpenCV, MediaPipe, and Streamlit")

# Webcam input
st.sidebar.header("Settings")
use_webcam = st.sidebar.checkbox("Use Webcam", True)

# Video Capture
if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        cap = cv2.VideoCapture(uploaded_file.name)
    else:
        st.warning("Please upload a video or enable the webcam.")
        st.stop()

# Create an output placeholder
frame_window = st.empty()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    results = pose.process(frame)

    # Draw keypoints
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display result
    frame_window.image(frame, channels="RGB")

# Release resources
cap.release()

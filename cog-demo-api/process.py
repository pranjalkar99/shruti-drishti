from cog import BasePredictor, Input, Path
import os
from typing import List
import cv2
import numpy as np
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# import profanity_check
from PIL import Image









def extract_pose_landmarks(frame, pose):
    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = pose.process(rgb_frame)

    # Extract landmarks if available
    if results.pose_landmarks:
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
        return landmarks
    else:
        return None




# Main function to process video files and assign labels
def process_videos(root_folder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    dataset = []
    labels = []

    sign_folders = os.listdir(root_folder)
    print(sign_folders)
    for sign_folder in sign_folders:
        sign_path = os.path.join(root_folder, sign_folder)
        print(sign_path)
        if os.path.isdir(sign_path):
            for filename in os.listdir(sign_path):
                print(filename)
                if filename.lower().endswith(('.mov', '.mp4')):
                    filepath = os.path.join(sign_path, filename)
                    print(filepath)
                    # Open video file
                    cap = cv2.VideoCapture(filepath)

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        landmarks = extract_pose_landmarks(frame, pose)
                        if landmarks is not None:
                            dataset.append(landmarks)
                            labels.append(sign_folder)  # Assign label based on folder name

                    cap.release()

    # Convert dataset and labels to numpy arrays
    dataset = np.array(dataset)
    labels = np.array(labels)

    # Save dataset and labels
    np.save('pose_landmarks_dataset.npy', dataset)
    np.save('pose_landmarks_labels.npy', labels)

    # Clean up
    pose.close()

    return dataset, labels

def postprocess(output):
    return str(output)


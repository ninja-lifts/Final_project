import os  # Module for interacting with the operating system
import pickle  # Module for serializing and deserializing Python objects . Make dataset continous 

import mediapipe as mp  # Library for processing hands, face, etc.
import cv2  # Library for computer vision tasks
import matplotlib.pyplot as plt  # Library for plotting (not used here but imported)

# Initialize Mediapipe hands module with specified settings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # For drawing styles

# Configure the Hands model with static image mode and a minimum detection confidence threshold
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory where the image dataset is stored
DATA_DIR = './Downloads/data'

# Initialize empty lists to store feature data and their corresponding labels
data = []
labels = []

# Iterate through each subdirectory in the dataset directory
for dir_ in os.listdir(DATA_DIR):
    # Iterate through each image in the current subdirectory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to hold the processed features of one image

        x_ = []  # List to store x-coordinates of hand landmarks
        y_ = []  # List to store y-coordinates of hand landmarks

        # Read the image using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convert the image from BGR to RGB (Mediapipe uses RGB images)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:  # Check if any hand landmarks are detected
            for hand_landmarks in results.multi_hand_landmarks:  # Iterate through detected hands
                # Extract x and y coordinates of each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)  # Append x-coordinate to the list
                    y_.append(y)  # Append y-coordinate to the list

                # Normalize the coordinates relative to the bounding box
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize x-coordinate
                    data_aux.append(y - min(y_))  # Normalize y-coordinate

            # Append the processed data and its label to the main lists
            data.append(data_aux)
            labels.append(dir_)  # Label is the name of the subdirectory (e.g., 'a', 'b', etc.)

# Save the collected data and labels to a pickle file
f = open('data.pickle', 'wb')  # Open a file in write-binary mode
pickle.dump({'data': data, 'labels': labels}, f)  # Serialize the data and labels into the file
f.close()  # Close the file

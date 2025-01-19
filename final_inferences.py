import pickle  
import cv2  
import mediapipe as mp  
import numpy as np  

# Load the pre-trained model stored in 'model.p'
model_dict = pickle.load(open('./Downloads/model_1.p', 'rb'))  # Load model from pickle file
model = model_dict['model_1'] 

cap = cv2.VideoCapture(0)  # Open the webcam, specify the camera device index

mp_hands = mp.solutions.hands  
mp_drawing = mp.solutions.drawing_utils  
mp_drawing_styles = mp.solutions.drawing_styles 

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) 

labels_dict = {
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T',
    'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z'
}


while True:
    data_aux = []  
    x_ = []  
    y_ = []  

    ret, frame = cap.read() 
    H, W, _ = frame.shape 
    # Convert the captured frame from BGR to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
  
    results = hands.process(frame_rgb) 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),  
                mp_drawing_styles.get_default_hand_connections_style() 
            )

        # Loop through the detected landmarks to extract the normalized x and y coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x 
                y = hand_landmarks.landmark[i].y 

                x_.append(x)
                y_.append(y)

           
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x 
                y = hand_landmarks.landmark[i].y 
              
                data_aux.append(x - min(x_))  
                data_aux.append(y - min(y_))  

        x1 = int(min(x_) * W) - 10  
        y1 = int(min(y_) * H) - 10  

        x2 = int(max(x_) * W) - 10  
        y2 = int(max(y_) * H) - 10  

        # Predict the letter corresponding to the hand gesture using the trained model
        prediction = model.predict([np.asarray(data_aux)]) 

        # Map the predicted numerical label to its corresponding character (e.g., 'A', 'B', 'L')
        predicted_character = labels_dict[int(prediction[0])]  

        # Draw a rectangle around the hand and display the predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA) 

    # Display the processed frame with the landmarks, rectangle, and predicted character
    cv2.imshow('frame', frame)  # Show the frame with drawing on the screen
    cv2.waitKey(1)  # Wait for 1 ms for a key press to continue the loop

# Release the webcam and close all OpenCV windows when the loop ends
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows

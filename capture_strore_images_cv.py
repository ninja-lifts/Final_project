import os
import cv2

# Directory where the dataset will be stored
DATA_DIR = './Downloads/data'
# Create the directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (26 for the English alphabet a-z)
number_of_classes = 26
# Number of images to collect per class
dataset_size = 500

# Initialize the webcam (camera index 0)
cap = cv2.VideoCapture(0)

# Loop through all classes (letters 'a' to 'z')
for j in range(number_of_classes):
    # Convert class index to corresponding letter ('a' for 0, 'b' for 1, etc.)
    letter = chr(97 + j)
    # Create a directory for the current class if it doesn't already exist
    class_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Step 1: Wait for user readiness before starting data collection
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        # Display a message prompting the user to press 'Q' to start
        cv2.putText(frame, 'Ready ? Press "Q" ', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        # Show the current frame
        cv2.imshow('frame', frame)
        # Check if the user presses 'Q' to start data collection
        if cv2.waitKey(1) == ord('q'):
            break

    # Step 2: Collect images for the current class
    counter = 0
    while counter < dataset_size:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        # Show the current frame
        cv2.imshow('frame', frame)
        # Wait 25 milliseconds between frames
        cv2.waitKey(25)
        # Save the captured frame to the dataset folder with a unique filename
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        # Increment the counter
        counter += 1

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

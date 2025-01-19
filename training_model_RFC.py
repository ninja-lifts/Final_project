import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset from the 'data.pickle' file using pickle
# The file contains a dictionary with 'data' (features) and 'labels' (classes)
data_dict = pickle.load(open('./Downloads/data.pickle', 'rb'))

# Convert the 'data' and 'labels' from the loaded dictionary to numpy arrays for further processing
data = np.asarray(data_dict['data'])  # Features array
labels = np.asarray(data_dict['labels'])  # Corresponding class labels

# Split the dataset into training and testing subsets
# `test_size=0.2` means 20% of the data will be used for testing, and 80% for training
# `shuffle=True` randomizes the dataset before splitting
# `stratify=labels` ensures the class distribution remains consistent in both sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier, a robust machine learning model
model_1 = RandomForestClassifier()

# Train the model using the training data (features and labels)
model_1.fit(x_train, y_train)

# Use the trained model to make predictions on the test data
y_predict = model_1.predict(x_test)

# Evaluate the model's performance by comparing predictions with the actual labels
# `accuracy_score` calculates the percentage of correctly classified samples
score = accuracy_score(y_predict, y_test)

# Print the accuracy of the model in percentage form
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a file named 'model.p' using pickle
# The model is saved as a dictionary with the key 'model'
f = open('model_1.p', 'wb')  # Open file in write-binary mode
pickle.dump({'model': model_1}, f) 
f.close()  

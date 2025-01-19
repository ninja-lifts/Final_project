# Import required libraries
import pickle
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./Downloads/data.pickle', 'rb'))

data = np.asarray(data_dict['data'])  
labels = np.asarray(data_dict['labels'])  

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5) 

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained KNN model to a file named 'model_knn.p' using pickle
f = open('model_knn.p', 'wb')  # Open file in write-binary mode
pickle.dump({'model': model_3}, f)  # Serialize and save the model
f.close()  

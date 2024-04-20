import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

#dataset import
dataset = pd.read_csv('./data/train.csv')

X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values

# Save column names for later use
feature_names = dataset.columns[:20]
label_name = dataset.columns[20]  

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
# print('Normalized data:')

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
print('One hot encoded array:')
print(y[0])
from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

# Test data does not have labels, getting test data to use as unseen here.

# Extract feature names for the one-hot encoded labels
label_names = ohe.get_feature_names(input_features=[label_name])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create DataFrames for the test data using the original column names
test_features_df = pd.DataFrame(X_test, columns=feature_names)
test_labels_df = pd.DataFrame(y_test, columns=label_names)

# Concatenate the feature and label DataFrames
test_df = pd.concat([test_features_df, test_labels_df], axis=1)

# Save the DataFrame to a CSV file
test_df.to_csv('./data/unseen.csv', index=False)

# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Get test data here: the old one was saved as unseen, X_t is now train data and x_v is for testing
X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.1)

history = model.fit(X_t, y_t, epochs=100, batch_size=64)

model.save('./model/MP2.h5')

#testing model
y_pred = model.predict(X_v)

#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_v)):
    test.append(np.argmax(y_v[i]))


a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
print( len(model.layers))

#Re initialized to delete trained weights
# Neural network
# model = Sequential()
# model.add(Dense(16, input_dim=20, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(4, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:18:21 2020

@author: VíctorLluva
"""

# mlp for binary classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# load the dataset

df = pd.read_csv("C:\Games\python_test.csv")

# split into input and output columns
X, y = df.values[:, 0:9], df.values[:, 10]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer 
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=2)
# evaluate the model (Accuracy)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
# evaluate the model (Recall)
from sklearn.metrics import recall_score
recall_score(y, predictions, average='weighted')

predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(200):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
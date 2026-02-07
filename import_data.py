import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Importing and setting up data for models
df = pd.read_csv('churn.csv')
x = pd.get_dummies(df.drop(['Churn', 'CustomerID'], axis=1))

# churn col is 0, 1 0 which can be easily passed into numpy
y = df['Churn'].to_numpy(dtype=np.float32)

# Used for new_predictions
original_columns = x.columns
with open('columns.pkl', 'wb') as f:
    pickle.dump(original_columns, f)
    

# need to convert to a numpy readable datatype
X = x.to_numpy(dtype=np.float32)


# Setting and training test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Scales data to be relative and similairly contribute to data
# BEFORE : 0 < age < 100, 20k < salary < 200k
# AFTER : 0 < age < 1, 0 < salary < 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open('scalar.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# Import for models
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from sklearn.metrics import accuracy_score

# Builds our Neural Network
#
# Input is the first layer of our NN and converts 
# raw features to data thats easier for the second layer to work with
# First Dense works with that data with 64 neurons or 64 different patterns
# then passes data into second dense
# Second Dense acts as as the return for our output value
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


# Complies out model
# TLDR
#
# loss = sum of how off the estimations are from the actual answer
# optimizer = value to change based on how far off loss is
# metrics = evaluates how well the model is performing
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting, predicting and evaluating data
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))

# Converts data to 1 or 0 based on test
predicts = (model.predict(X_test) > 0.5).astype(int)
# Returns the accuracy score for all predicts
print("Accuracy: ", accuracy_score(y_test, predicts))

# Save the model on disk
model.save('churn_model.keras')


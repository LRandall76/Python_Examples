# Seed value
seed_number = 1

# 1. Set Seed for PYTHONHASHSEED
import os
os.environ['PYTHONHASHSEED']=str(seed_number)

# 2. Set Seed for Python
import random
random.seed(seed_number)

# 3. Set Seed for Numpy
import numpy as np
np.random.seed(seed_number)

# 4. Set Seed for Tensorflow
import tensorflow as tf
tf.random.set_seed(seed_number)

# Matplotlib for Plotting
import matplotlib.pyplot as plt

# Pandas for Extracting and Manipulating Matrices
import pandas as pd
from pandas import DataFrame
from pandas import concat

# LSTM Layers
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# Saving Model Functions
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Splitting and Normalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

######## Next For RNN #3
target = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/load.txt', sep=" ", header=None)
matrix = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/matrix.txt, header=None)

#Scale the all of the data to be values between 0 and 1 
mscaler = MinMaxScaler(feature_range=(0, 1)) 
matrix = mscaler.fit_transform(matrix)
tscaler = MinMaxScaler(feature_range=(0, 1)) 
target = tscaler.fit_transform(target)

## Set Range
# 2 Weeks
x_test = matrix[38897:39233]
y_test = target[38897:39233]

x_train = matrix[0:38896]
y_train = target[0:38896]

# 1 Year
#x_test = matrix_scaled[30481:39233]
#y_test = target_scaled[30481:39233]

#x_train = matrix_scaled[8662:30481]
#y_train = target_scaled[8662:30481]

# Reshape to a 3D Array
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# Create RNN Model
rnn_model = Sequential() #initializing sequential model, layers can be added sequentially with model.add

# Add 3 Hidden Layers
rnn_model.add(SimpleRNN(30, input_shape=(x_train.shape[1], 1), return_sequences=True)) #simple recurrent layer, 30 neurons & process 10x1 sequences
rnn_model.add(SimpleRNN(30,  return_sequences=True)) #simple recurrent layer, 30 neurons & process 10x1 sequences
rnn_model.add(SimpleRNN(30,  return_sequences=False)) #simple recurrent layer, 10 neurons & process 10x1 sequences
# Add 2 Dense Hidden Layer for Ouptut
rnn_model.add(Dense(8,activation='tanh')) #Dense layer, 8 neurons w/ tanh activation
rnn_model.add(Dense(1, activation='linear')) #linear output because this is a regression problem

# Complie the Model
rnn_model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

# Train the Model
history = rnn_model.fit(x_train,y_train,epochs=50)

# Extract Accuracy
_, accuracy = rnn_model.evaluate(x_train, y_train)
# check predicted values
predictions = rnn_model.predict(x_test) 
# Undo scaling
predictions = tscaler.inverse_transform(predictions)
y_test = tscaler.inverse_transform(y_test)

# Calculate RMSE score
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

# Calculate MAPE
mape = np.mean(np.abs(y_test - predictions)/y_test) * 100
print(mape)

# 2 Weeks
plt.plot(y_test,'k')
plt.plot(predictions,'r')

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

# Plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#Parameter Optimization
from hyperopt import hp

# For Math Stuff
import math

######## Format and Partition Data
target = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/load.txt', sep=" ", header=None)
matrix = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/matrix.txt', header=None)

# Scale All of the Data to be Values Between 0 and 1 
mscaler = MinMaxScaler(feature_range=(0, 1)) 
matrix = mscaler.fit_transform(matrix)

tscaler = MinMaxScaler(feature_range=(0, 1)) 
target = tscaler.fit_transform(target)

matrix = matrix[8661:len(matrix)-1]
target = target[8661:len(target)-1]

## Set Range
# 30 %
x_test = matrix[math.floor(len(matrix) * 0.7):(len(matrix)-1)]
y_test = target[math.floor(len(target) * 0.7):(len(target)-1)]

x_train = matrix[0:math.floor(len(matrix) * 0.7 - 1)]
y_train = target[0:math.floor(len(target) * 0.7 - 1)]

# Declare A Simple ANN
ann_model = Sequential()

# Add First Hidden Layer
ann_model.add(Dense(40, input_dim=x_train.shape[1], activation='selu')) # Comment Or Uncomment to add ANN layer

# Uncomment to Add an Additional Layer
#ann_model.add(Dense(40, activation='selu'))

# Output Layer
ann_model.add(Dense(1, activation='sigmoid'))

# Compile
ann_model.compile(loss= tf.keras.losses.MeanAbsolutePercentageError(), optimizer='adam', metrics=['accuracy'])

# Used for Validation Stopping and Saving the Best Performing Network
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('ann_best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# Fit the Keras Model on the Dataset
history = ann_model.fit(x_train, y_train, epochs = 1000, batch_size = 128, validation_split = 0.3, verbose=1, callbacks=[es, mc])

# Load In Best Model For Testing
ann_model = load_model('ann_best_model.h5')

_, accuracy = ann_model.evaluate(x_train, y_train)
# Check Predicted Values
predictions = ann_model.predict(x_test) 
# Undo Scaling
predictions = tscaler.inverse_transform(predictions)
y_test = tscaler.inverse_transform(y_test)

# Calculate RMSE Score
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

# Calculate MAPE
mape = np.mean(np.abs(y_test - predictions)/y_test) * 100
print(mape)

# Plot Data
# For 1 Year
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.plot(y_test/1000,'k',label = "Actual")
plt.plot(predictions/1000,'r', label = "Predicted")
plt.title('ANN Prediction vs Actual 2019')
plt.xlabel('Time (h)')
plt.ylabel('Load Value (GW)')
plt.legend()
plt.grid()

# Summarize History for Loss
plt.plot(history.history['loss'],'k')
plt.plot(history.history['val_loss'],'r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()

# List All Data in History
print(history.history.keys())

# Summarize History for Accuracy
plt.plot(history.history['accuracy'],'k')
plt.plot(history.history['val_accuracy'],'r')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
plt.show()

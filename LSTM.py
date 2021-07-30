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

# Parameter Optimization
import hyperopt as hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Import Variables
target = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/Load/load.txt', sep=" ", header=None)
matrix = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/Hour Ahead/matrix.txt', header=None)

# Seperate Previous Load Columns
load1 = matrix[8]
load2 = matrix[9]

# Append the Columns Together with Target
temp = np.append(load1,load2)
loadall = np.append(temp,target)
loadall = pd.DataFrame(loadall, columns = ['1'])

# Normalize the Columns with Respect to the Highest Value
# Scale the all of the data to be values between 0 and 1 
lscaler = MinMaxScaler(feature_range=(0, 1)) 
loadall = lscaler.fit_transform(loadall)
tscaler = MinMaxScaler(feature_range=(0, 1)) 
target  = tscaler.fit_transform(target)
mscaler = MinMaxScaler(feature_range=(0, 1)) 
matrix = mscaler.fit_transform(matrix)

# Replace Matrix Values with New Normalized Values
matrix[:,8] = loadall[0:39243,0]
matrix[:,9] = loadall[39243:(39243 + 39243),0]
target = loadall[39243 + 39243:(39244 + 39243 * 2),0]

# Initialize Containers for Creating Time Series
windowSize, x_Temp, y_Temp = 30, [], [] #initialize lists and set window size

# Create Sequence using provided data
for index in range(len(matrix)-windowSize): #we must end at train-windowSize to avoid the windowSize going past the end
    if index > 8661: # Remember the First Year of your Data is Garbage so Don't use It!
        temp = matrix[index:index+windowSize]
        temp = np.transpose(temp)
        x_Temp.append(temp) #append the range from index to index+windowSize to x
        y_Temp.append(target[index-1+windowSize]) #append the next value to the y

# Convert to Numpy Array
x_Temp  = np.array(x_Temp)
y_Temp  = np.array(y_Temp)
    
# Save Values and Print to Test ----> Sequence Goes for windowSize = 3: Matrix[0-2,:] = Target[2]      
np.savetxt("3D.csv", x_Temp[0], delimiter=",")

## Partition Data
# 30% of Data as Testing Set about 1 Year
x_test  = x_Temp[math.floor(len(x_Temp) * 0.7):(len(x_Temp)-1)]
y_test  = y_Temp[math.floor(len(x_Temp) * 0.7):(len(x_Temp)-1)]
x_train = x_Temp[0:math.floor(len(x_Temp) * 0.7 - 1)]
y_train = y_Temp[0:math.floor(len(x_Temp) * 0.7 - 1)]

# Forecast for 2 Weeks
#x_test  = x_Temp[(len(x_Temp)-337):(len(x_Temp)-1)]
#y_test  = y_Temp[(len(x_Temp)-337):(len(x_Temp)-1)]
#x_train = x_Temp[0:(len(x_Temp)-338)]
#y_train = y_Temp[0:(len(x_Temp)-338)]

# Reset Objects To Numpy Arrays
x_test  = np.array(x_test)
y_test  = np.array(y_test)
x_train = np.array(x_train)
y_train = np.array(y_train)

###### Initialize LSTM 
# Declare Sequential Model
lstm_model = Sequential()

# Initialize Layer: Remember to Change Returne Sequences = False for Last Regression Layer!!!! and for any in between layer use return_sequences = true
lstm_model.add(LSTM(units = 106, return_sequences = False, activation="selu", recurrent_activation="sigmoid", recurrent_dropout=0.0, unroll=False, use_bias=True, input_shape = (x_train.shape[1], x_train.shape[2])))
lstm_model.add(Dropout(0.283))

# Adding the output layer
lstm_model.add(Dense(1, activation='linear')) #linear output because this is a regression problem

#Build Model with Loss Function
lstm_model.compile(optimizer='adam', loss= tf.keras.losses.MeanAbsolutePercentageError(), metrics = ['accuracy'])

# Used for Validation Stopping and Saving the Best Performing Network
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# Output a Summary of the Model
lstm_model.summary()

# Fit The Model
history = lstm_model.fit(x_train, y_train, epochs = 1400, batch_size = 100, validation_split = 0.3, verbose=1, callbacks=[es, mc])

# Replace Current Model with Best Performing Model Over Training Set
lstm_model = load_model('best_model.h5')

# Check predicted values
predictions = lstm_model.predict(x_test) 

# Undo scaling
predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))
predictions = tscaler.inverse_transform(predictions)

y_test = np.reshape(y_test, (y_test.shape[0], 1))
y_test = tscaler.inverse_transform(y_test)

# Calculate RMSE score
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

# Calculate MAPE
mape = np.mean(np.abs(y_test - predictions)/y_test) * 100
print(mape)

# Plot Data
# For 1 Year
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.plot(y_test/1000,'k', label = "Actual")
plt.plot(predictions/1000,'r',label = "Predicted")
plt.title('LSTM Prediction vs Actual 2019')
plt.xlabel('Time (h)')
plt.ylabel('Load Value (GW)')
plt.legend(loc="upper right")
plt.grid()

# summarize history for loss
plt.plot(history.history['loss'],'k')
plt.plot(history.history['val_loss'],'r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()


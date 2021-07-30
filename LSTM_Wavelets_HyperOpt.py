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

# Pandas for Extracting and Manipulating Matrices
import pandas as pd

# Matplotlib for Plotting
import matplotlib.pyplot as plt

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

# Parameter Optimization
import hyperopt as hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

### Load in Variables and Formatting
# Import Variables
target = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/Load/load_wave_4_db10.csv', header=None)
matrix = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/Hour Ahead/matrix_hour_ahead_wave_4_db10.csv', header=None)

# Load In Arrays Without Wavelets
wotarget = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/Load/load.csv', header=None)
womatrix = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/Hour Ahead/matrix.csv', header=None)

## Seperate Previous Load Columns
# Approximation Level
load1 = matrix[8]
load2 = matrix[13]
load3 = target[0]

# Append the Columns Together with Target
temp = np.append(load1,load2,axis=0)
loadall = np.append(temp,load3,axis=0)

# Expand Dimensions of Arrays to Help with Concatonation
load1   = np.expand_dims(load1,axis=0)
load2   = np.expand_dims(load2,axis=0)
load3   = np.expand_dims(load3,axis=0)
temp    = np.expand_dims(temp,axis=0)
loadall = np.expand_dims(loadall,axis=0)

### Do the Same for the Arrays Without Wavelets
## Seperate Previous Load Columns
# Approximation Level
load4 = womatrix[8]
load5 = womatrix[9]
load6 = wotarget[0]

# Append the Columns Together with Target
temp = np.append(load4,load5,axis=0)
woloadall = np.append(temp,load6,axis=0)

# Expand Dimensions of Arrays to Help with Concatonation
load4   = np.expand_dims(load4,axis=0)
load5   = np.expand_dims(load5,axis=0)
load6   = np.expand_dims(load6,axis=0)
temp    = np.expand_dims(temp,axis=0)
woloadall = np.expand_dims(woloadall,axis=0)

# Find Level of Decomp
level = (target.shape)

# Detail Levels
for i in range(level[1] - 1):
    # Extract Columns
    load1 = matrix[9+i]
    load2 = matrix[14+i]
    load3 = target[1+i]
    # Append Columns Vertically
    temp     = np.append(load1,load2,axis=0)
    loadtemp = np.append(temp,load3,axis=0)
    loadtemp = np.expand_dims(loadtemp,axis=0)
    # Append Horizontally
    loadall  = np.append(loadall,loadtemp,axis=0)
    
# Add on Without Wavelet Arrays
loadall  = np.append(loadall,woloadall,axis=0)
# Convert to Panda's DataFrame
loadall = pd.DataFrame(loadall)     
loadall = loadall.T    
# Reload in target for Predicting
target = pd.read_csv('/Users/retribuxion/desktop/resources/Data File/Load/load.csv', header=None)
# Initialize an Empty Object to Put Varaibles Into
matrixall = np.zeros([39243, 20])

# Normalize the Columns with Respect to the Highest Value
# Scale the all of the data to be values between 0 and 1 
lscaler = MinMaxScaler(feature_range=(0, 1)) 
loadall = lscaler.fit_transform(loadall)
tscaler = MinMaxScaler(feature_range=(0, 1)) 
target  = tscaler.fit_transform(target)
mscaler = MinMaxScaler(feature_range=(0, 1)) 
matrix = mscaler.fit_transform(matrix)

# Replace Matrix Values with New Normalized Values
for i in range(level[1] + 1):
    matrixall[:,8+i] = loadall[0:39243,i]
    matrixall[:,14+i] = loadall[39243:(39243 + 39243),i]
# Continue Replacing   
for i in range(8):        
    matrixall[:,i] = matrix[:,i]
        
# Initialize Containers for Creating Time Series
windowSize, x_Temp, y_Temp = 30, [], [] #initialize lists and set window size

# Create Sequence using provided data
for index in range(len(matrixall)-windowSize): # We must end at train-windowSize to avoid the windowSize going past the end
    if index > 8661: # The First Year of my Data Hurts Results so I Remove It Here
        temp = matrixall[index:index+windowSize]
        temp = np.transpose(temp)
        x_Temp.append(temp) # Append the range from index to index+windowSize to x
        y_Temp.append(target[index-1+windowSize]) # Append the next value to the y

# Convert to Numpy Array
x_Temp  = np.array(x_Temp)
y_Temp  = np.array(y_Temp)
   
## Partition Data
# 30% of Data as Testing Set about 1 Year
x_test       = x_Temp[math.floor(len(x_Temp) * 0.7):(len(x_Temp)-1)]
y_test       = y_Temp[math.floor(len(x_Temp) * 0.7):(len(x_Temp)-1)]
x_train      = x_Temp[0:math.floor(len(x_Temp) * 0.7 - 1)]
y_train      = y_Temp[0:math.floor(len(x_Temp) * 0.7 - 1)]
    

# Reset Objects To Numpy Arrays
x_test  = np.array(x_test)
y_test  = np.array(y_test)
x_train = np.array(x_train)
y_train = np.array(y_train)

### Save For Recalling During HyperOpt Loop
# Reshape to 2D Array from 3D
x_test_reshaped  = x_test.reshape(x_test.shape[0], -1) 
x_train_reshaped = x_train.reshape(x_train.shape[0], -1) 

# Save 2D Array
np.savetxt("/Users/retribuxion/desktop/resources/Storage_For_Train_Test/x_test_reshaped.txt", x_test_reshaped, delimiter=",")
np.savetxt("/Users/retribuxion/desktop/resources/Storage_For_Train_Test/y_test.txt", y_test, delimiter=",")
np.savetxt("/Users/retribuxion/desktop/resources/Storage_For_Train_Test/x_train_reshaped.txt", x_train_reshaped, delimiter=",")
np.savetxt("/Users/retribuxion/desktop/resources/Storage_For_Train_Test/y_train.txt", y_train, delimiter=",")

# Transfer 3rd Shape of Array to New Cell
x_test_shape2  = x_test.shape[2]
x_train_shape2 = x_train.shape[2]
x_test_shape2  = np.array(x_test_shape2)
x_train_shape2 = np.array(x_train_shape2)
x_test_shape2  = x_test_shape2.reshape(1) 
x_train_shape2 = x_train_shape2.reshape(1) 

# Save Last Dimension of Original Array for Reference Later
np.savetxt("/Users/retribuxion/desktop/resources/Storage_For_Train_Test/x_test_shape2.txt", x_test_shape2, delimiter=",")
np.savetxt("/Users/retribuxion/desktop/resources/Storage_For_Train_Test/x_train_shape2.txt", x_train_shape2, delimiter=",")

# Define a Search Space for HyperOpt
space = {'choice': hp.choice('num_layers',
                    [ {'layers':'one', },
                      {'layers':'two',
                    'units2'   : hp.uniform('units2', 64,128)}
                    'dropout2' : hp.uniform('dropout2', .25,.75)}
                      ]),

            'units1'  : hp.uniform('units1', 64,128),
            'dropout1': hp.uniform('dropout1', .25,.75),
            'optimizer'   : 'adam',
            'activation'  : hp.choice('activation',['relu','sigmoid','tanh','selu'])
         
            #### Additional Parameters to Optimize For, If Desired 
            #'batch_size' : hp.uniform('batch_size', 28,128),
            #'epochs'     : hp.uniform('epochs',  50, 200),
            #'patience'    : hp.uniform('patience', 25, 100),
            #'optimizer'   : hp.choice('optimizer',['adadelta','adam','adamax','adagrad','ftrl','nadam','rmsprop','sgd']), # This is ALL of the Optimizers
            #'activation'  : hp.choice('activation',['relu','sigmoid','softmax','softplus','softsign','tanh','selu','elu','exponential']) # This is ALL of the Activation Functions   
        }

# Define an Objective Function to Optimized within HyperOpt Functions
def objective(params):
   
    # Set Seed Values
    seed_value = 1
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    # Load Reshaped X and Regular Y Values from Save File
    x_test_reshaped  = pd.read_csv('/Users/retribuxion/desktop/resources/Storage_For_Train_Test/x_test_reshaped.txt',  header=None)
    y_test           = pd.read_csv('/Users/retribuxion/desktop/resources/Storage_For_Train_Test/y_test.txt',  header=None)
    x_train_reshaped = pd.read_csv('/Users/retribuxion/desktop/resources/Storage_For_Train_Test/x_train_reshaped.txt',  header=None)
    y_train          = pd.read_csv('/Users/retribuxion/desktop/resources/Storage_For_Train_Test/y_train.txt',  header=None)

    # Load Shape of Last Dimensions
    x_test_shape2    = pd.read_csv('/Users/retribuxion/desktop/resources/Storage_For_Train_Test/x_test_shape2.txt', header=None)
    x_train_shape2   = pd.read_csv('/Users/retribuxion/desktop/resources/Storage_For_Train_Test/x_train_shape2.txt', header=None)

    # Reshape Data to Old Shapes
    x_test_shape2    = np.array(x_test_shape2)
    x_test_reshaped  = np.array(x_test_reshaped)
    x_test = x_test_reshaped.reshape(x_test_reshaped.shape[0], x_test_reshaped.shape[1] // x_test_shape2[0,0].astype(int), x_test_shape2[0,0].astype(int))
    x_train_shape2   = np.array(x_train_shape2)
    x_train_reshaped = np.array(x_train_reshaped)
    x_train = x_train_reshaped.reshape(x_train_reshaped.shape[0], x_train_reshaped.shape[1] // x_train_shape2[0,0].astype(int), x_train_shape2[0,0].astype(int))
    
    ###### Initialize LSTM 
    # Declare Sequential Model
    lstm_model = Sequential()
    
    # Check if return_sequences parameters needs to be true or false
    if params['choice']['layers']== 'two':
            seq_return = True
    else:
            seq_return = False
            
    # Round Down Units
    units1 = math.floor(params['units1'])
    
    # Initialize Layer: Remember to Change Returne Sequences = False for Last Regression Layer!!!! and for any in between layer use return_sequences = true
    lstm_model.add(LSTM(units = units1, return_sequences = seq_return, activation=params['activation'], recurrent_activation="sigmoid", recurrent_dropout=0.0, unroll=False, \
                        use_bias=True, input_shape = (x_train.shape[1], x_train.shape[2])))
    lstm_model.add(Dropout(params['dropout1'])
    
    
    if params['choice']['layers']== 'two':
    # Adding a second LSTM layer and Dropout layer
        units2 = math.floor(params['choice']['units2'])
        lstm_model.add(LSTM(units = units2, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0.0, unroll=False, use_bias=True, return_sequences = False))
        lstm_model.add(Dropout(params['choice']['dropout2']))
                
    # Adding the output layer
    lstm_model.add(Dense(1, activation='linear')) 

    #Build Model with Loss Function
    lstm_model.compile(optimizer=params['optimizer'], loss= tf.keras.losses.MeanAbsolutePercentageError(), metrics = ['accuracy'])
    
    # Used for Validation Stopping and Saving the Best Performing Network
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
  
    # Display Summary of Model
    lstm_model.summary()
    # Train the Model
    history = lstm_model.fit(x_train, y_train, epochs = 1000, batch_size = 128, validation_split = 0.3, verbose=0,  callbacks=[es, mc])
                          
    ## Predictions        
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
    #print(rmse) # Optional Root Mean Square Accuracy Parameter

    # Calculate MAPE
    mape = np.mean(np.abs(y_test - predictions)/y_test) * 100
    clear_output(wait=True) 
    print('Mape: ' + str(mape))
    return mape
  
# Minimize the Objective Over the Space
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=12, trials = trials)

# Show Best Result
print('Best: ' + str(best))

#print(trials.trials) # To View Results of HyperOpt with All Details

# Display All Results
print(trials.losses())

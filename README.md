# Python_Examples
Examples of My Code Writing Ability in Python


## ANN.py
Imports and formats data. Then creates and trains a Simple ANN to generate results with plots.  

## ARIMA.py
Imports and formats data. Generates several plots to identify relative parameters for an ARIMA network. Then, an ARIMA network is
created and tested for accuracy. Also, plots results.

## LSTM.py 
This example formats data for an LSTM network by normalizing the input data and partitioning the data to a training and testing set.
From here, a LSTM structure is created and trained using the partitioned training set data. Finally, the testing set data is used to 
create results for the network which are plotted and compared with actual results for the timeframe for accuracy using MAPE (Mean Absolute 
Percentage Error).

## LSTM_Wavelets_HyperOpt.py
Very similar to LSTM.py but with more data formating and then defining of a function and search space for optimization. Plots results.

## RNN.py
Almost Identical to ANN code but with data sequenced into a "time series."

## Tracking_Thermal_Camera_Code.py
A snippit of code that was used to track a thermal body for my Senior Design Project. Works with a Raspberry Pi 4, a FLIR Lepton 3.5 Camera, 
and a set of servos w/ servo driver.

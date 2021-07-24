# Pandas for Importting Files
import pandas as pd
# Matplotlib for Plotting
from matplotlib import pyplot
import matplotlib.pyplot as plt

import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose    
import statistics
from pmdarima.arima.utils import ndiffs

# Numpy for MEAN Calculation
import numpy as np
# For Organizing Datetime Variables
from datetime import datetime

### Read In Data
rawdata = pd.read_csv('/Users/retribuxion/desktop/resources/LOADMaster.txt', sep=" ", header=None)
matrix = pd.read_csv('/Users/retribuxion/desktop/resources/MATRIXMaster.txt', sep=" ", header=None)
date = pd.read_csv('/Users/retribuxion/desktop/resources/datetime_vector.csv', header=None)
data = pd.read_csv('/Users/retribuxion/desktop/resources/datetime_and_data.csv', header=None)

# Format Data
data.columns = ["Date","Load"]
data.drop(39262, axis = 0, inplace = True)

# Organize Dates
data.set_index('Date', inplace = True)
data.describe().transpose()
index = pd.date_range('02/11/2015', periods=39277, freq='H')
adjustedindex = index[15:39277]
data.set_index(adjustedindex, inplace=True) 
data_series = data["Load"]

# Set Frequency
data.asfreq(freq = '1h')

### Plots of Decompositions of Data
decomp = seasonal_decompose(data_series, period = 8648)
fig = decomp.plot()

# Plot to Find D Parameter
fig, axes = plt.subplots(4, 2, sharex = False)

# Original Series
axes[0,0].plot(rawdata); axes[0,0].set_title('Original Series')
plot_acf(rawdata, ax=axes [0,1])

# 1st Differencing 
axes[1,0].plot(rawdata.diff()); axes[1,0].set_title('1st Order Differencing')
plot_acf(rawdata.diff().dropna(), ax=axes[1,1])

# 2nd Differencing
axes[2,0].plot(rawdata.diff().diff()); axes[2,0].set_title('2nd Order Differencing')
plot_acf(rawdata.diff().diff().dropna(), ax = axes[2,1])

# 3rd Differencing
axes[3,0].plot(rawdata.diff().diff().diff()); axes[3,0].set_title('3rd Order Differencing')
plot_acf(rawdata.diff().diff().diff().dropna(), ax = axes[3,1])

# Show Plots
plt.rcParams['figure.figsize'] = [30, 50]
plt.show()

# Plot to Find Q Paramater
fig, axes = plt.subplots(1, 2, sharex=False)
axes[0].plot(rawdata.diff()); axes[0].set_title('1st Differencing')
plot_pacf(rawdata.diff().dropna(), ax=axes[1])
plt.rcParams['figure.figsize'] = [40, 20]
plt.show()

### Begin Fitting Model
# Code Referenced from https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
model = ARIMA(data_series[:38033], order=(2, 1, 2))  
fitted = model.fit(disp=-1) 

# Forecast
fc, se, conf = fitted.forecast(4336, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=data_series[34034:38370].index)
lower_series = pd.Series(conf[:, 0], index=data_series[34034:38370].index)
upper_series = pd.Series(conf[:, 1], index=data_series[34034:38370].index)

# Plot
plt.figure(figsize=(24,5), dpi=300)
plt.plot(data_series[:38033], label='training')
plt.plot(data_series[38034:38370], label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.xlim(adjustedindex[34034], adjustedindex[38370])
plt.ylim(60000, 110000)
plt.show()

mape = np.mean(np.abs(data_series[38034:38370] - fc_series[4000:4336])/data_series[38034:38370]) * 100
print(mape)

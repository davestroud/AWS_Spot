# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('short_test.csv')
df.head()

df = df.rename(columns = {'Timestamp':'Date'})
df.head()



df.info()

df.head()


df.Date

plt.figure(figsize=(18,9))
df.plot()


df = df.sort_values('Date')



train_data = df[:1144]
test_data = df[1144:]



# Scale the data to be between 0 and 1
# When scaling remember! You normalize both test and train data with
# respect to training data
# Because you are not supposed to have access to the test dataset
scaler = MinMaxScaler()
train_data = train_data.values.reshape(-1,1)
test_data = test_data.values.reshape(-1,1)


train_data

# Train the Scaler with training data and smooth data
smoothing_window_size = 350
for di in range(0,1000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

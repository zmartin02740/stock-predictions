#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np

r = requests.get(
    'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MVIS&interval=1min&outputsize=full&apikey=L6IGN76QB01URQP1')
jsonR = r.json()
timeSeries = jsonR['Time Series (1min)']

print('timeSeries')

              
              
              
              


# In[4]:


train_x = []
train_y = []
for time in timeSeries:
    train_x.append(time)
    train_y.append(float(timeSeries[time]['4. close']))
    
train_y = np.array(train_y)
df = pd.DataFrame(train_y, columns=['close'])
df.index = train_x
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
print(df)


# In[5]:


df.shape


# In[6]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()


# In[7]:



dataset = df.values
training_data_len = math.ceil(len(dataset) * .8)
training_data_len


# In[8]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[9]:


train_data = scaled_data[0:training_data_len, :]
X_train = []
Y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    Y_train.append(train_data[i, 0])
    if i <= 61:
        print(X_train)
        print(Y_train)


# In[10]:


X_train, Y_train = np.array(X_train), np.array(Y_train)
Y_train


# In[11]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


# In[12]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[13]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[14]:


model.fit(X_train, Y_train, batch_size=1, epochs=1)


# In[15]:


test_data = scaled_data[training_data_len - 60: , :]
X_test = []
Y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])


# In[16]:


X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[17]:


predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
print(X_test)


# In[79]:


rmse = np.sqrt(np.mean(predictions - Y_test)**2)
rmse


# In[89]:


train = df[:training_data_len]
valid = df[training_data_len:]
valid['Predictions'] = predictions


# In[90]:


plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Pred'], loc='lower right')
plt.show()


# In[91]:


valid


# In[4]:


from tqdm.notebook import tqdm_notebook
for i in tqdm_notebook(range(5)):
    print(i)


# In[93]:


pip install tqdm


# In[ ]:





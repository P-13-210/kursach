import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = ('2')

data = pd.read_csv('./teach/all_stocks_5yr.csv')
print(data.shape)
print(data.sample(10))
data['date'] = pd.to_datetime(data['date'])
data.info()
data_name_company = (input("Сокращенное название компании "))

company_name = data_name_company
print(company_name)

data['date'] = pd.to_datetime(data['date'])
data.info()



plt.figure(figsize=(19,10))
for index, company in enumerate([company_name], 1):
  plt.subplot(1, 1, index)
  c = data[data['Name'] == company]
  plt.plot(c['date'], c['close'], c="r", label="close", marker=".")
  plt.plot(c['date'], c['open'], c="g", label="open", marker=".")
  plt.title(company)
  plt.legend()
  plt.tight_layout()





plt.show()


plt.figure(figsize=(19,10))
for index, company in enumerate([company_name], 1):
  plt.subplot(1, 1, index)
  c = data[data['Name'] == company]
  plt.plot(c['date'], c['volume'], c='purple', marker='.')
  plt.title(f"{company} Volume")
  plt.tight_layout()


plt.show()



name = data[data['Name'] == company_name]
prediction_range = name.loc[(name['date'] > datetime(2013,1,1))
 & (name['date']<datetime(2018,1,1))]
plt.figure(figsize=(19,10))
plt.plot(name['date'], name['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title(f"{company} Stock Prices")


plt.show()

close_data = name.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)



scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]


x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary



model.compile(optimizer='nadam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=10)

test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test=np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

mse = np.mean(((predictions - y_test) ** 2))
print("MSE", mse)
print("RMSE", np.sqrt(mse))

train = name[:training]
test = name[training:]
test['Predictions'] = predictions

plt.figure(figsize=(19,10))
plt.plot(train['date'], train['close'])
plt.plot(test['date'], test[['close', 'Predictions']])
plt.title(f'{company_name} Stock Close Price')
plt.xlabel('date')
plt.ylabel("close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()





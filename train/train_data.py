import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def read_data():
    data = pd.read_csv('../data/VIP_day.csv', index_col='date')
    print(data)
    return data.values

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)


# normalize the dataset
dataset = read_data()
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
datasetY = dataset[:, 0]
scaler2 = MinMaxScaler(feature_range=(0, 1))
datasetY = scaler2.fit_transform(datasetY)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

print(train)

# use this function to prepare the train and test datasets for modeling
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(trainX)
print(trainX.shape)
print(trainX[0][0])
# reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# print(trainX)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(1, input_shape=(5, 12)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=10, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# invert predictions

trainPredict = scaler2.inverse_transform(trainPredict)
trainY = scaler2.inverse_transform(trainY)
testPredict = scaler2.inverse_transform(testPredict)
testY = scaler2.inverse_transform(testY)

print(trainPredict)
print(trainY)

print(len(list(trainY)), len(trainPredict))
trainScore = math.sqrt(mean_squared_error(list(np.reshape(trainY, (len(trainPredict), 1))), list(trainPredict)))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(list(np.reshape(testY, (len(testPredict), 1))), list(testPredict)))
print('Test Score: %.2f RMSE' % (testScore))


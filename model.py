from operator import mod
import numpy as np
import pandas as pd
import os

import tensorflow as tf

path ='Data/'

etfs = os.scandir(path + 'ETFs/')
stocks = os.scandir(path + 'Stocks/')

#find the value after the amount of days
days=20
X_train = []
Y_train = []
X_test = []
Y_test = []

numTickers = 20
count = 0

for etf in etfs:
    data = pd.read_csv(path + 'ETFs/' + etf.name)
    training_size = int(len(data)*0.80)
    data_len = len(data)

    train, test = data[0:training_size],data[training_size:data_len]

    train = train.loc[:, ["Open"]].values
    test = test.loc[:, ["Open"]].values

    train = tf.keras.utils.normalize(train, axis=0, order=2)
    test = tf.keras.utils.normalize(test, axis=0, order=2)

    #find the value based on the previous 'days' amount of days
    days=20

    for i in range(days, len(train)):
        X_train.append(train[i-days:i])
        Y_train.append(train[i])
    
    for i in range(days, len(test)):
        X_test.append(test[i-days:i])
        Y_test.append(test[i])
    
    count = count + 1
    if count >= numTickers:
        break
    
# for i in range(0,5):
#     print(X_train[i], Y_train[i])
# print(np.shape(X_train), np.shape(Y_train))

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(50, activation="tanh", return_sequences = True),
    tf.keras.layers.Dropout(0.1),
    
    tf.keras.layers.SimpleRNN(50, activation="tanh", return_sequences = True),
    tf.keras.layers.Dropout(0.1),
    
    tf.keras.layers.SimpleRNN(50, activation="tanh", return_sequences = True),
    tf.keras.layers.Dropout(0.1),
    
    tf.keras.layers.SimpleRNN(50),
    tf.keras.layers.Dropout(0.1),
    
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError())

BATCH_SIZE = 32
history = model.fit(np.array(X_train), np.array(Y_train), epochs=10, batch_size=BATCH_SIZE)


predictions = model.predict(np.array(X_test))

for i in range(0,5):
    print("predicted price to be: ", predictions[i])
    print("real price: ", np.array(Y_test)[i])
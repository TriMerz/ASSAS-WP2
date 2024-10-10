import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

os.system('cls' if os.name == 'nt' else 'clear')

cache_file = 'data_cache.pkl'

# ========== LOAD CACHED DATA ==========
if os.path.exists(cache_file):
   with open(cache_file, 'rb') as f:
      df = pickle.load(f)
   print("\n", "Cache loaded successfully!")
else:
   # define an error

# Normalization of the 2233 features (dt-timestep excluded) made by using the training set
# ========== NORMALIZATION ==========
features = df.drop(columns=['dtcesar'])

scaler = MinMaxScaler()
if os.path.exists(MinMaxWeights):
   scaler = joblib.load(MinMaxWeights)
   print("\n", "MinMaxScaler weights loaded successfully!")
else:
   scaler.fit(features.iloc[:int(len(df)*0.7),:])  # int(len(df)*0.7) lengh of the training set
   joblib.dump(scaler, MinMaxWeights)

norm_df = pd.DataFrame(scaler.transform(features), columns=features.columns, index=df.index)
norm_df.insert(0, 'dtcesar', df['dtcesar'])

# Making windows of 10 time step + 1 label and a windowed time for the plot
# ===== WINDOWING =====
def df_to_X_y(df, window_size=3):
   time = df.index.to_numpy()
   df_as_np = df.to_numpy()
   T = []
   X = []
   y = []
   for i in range(len(df_as_np)-window_size):
      window = df_as_np[i:i+window_size]
      X.append(window)
      label = df_as_np[i+window_size, 1:]
      y.append(label)
      time_w = time[i:(i+window_size+1)]
      T.append(time_w)
   return np.array(X), np.array(y), np.array(T)

WINDOW_SIZE = 10
X, y, t = df_to_X_y(norm_df, WINDOW_SIZE)
print("\n", "Shape of the Windowed DataBase: ", X.shape, "\n", "Shape of the label: ",y.shape)
print(" Shape of the Windowed Time: ", t.shape, "\n")

n = len(X)
X_train, y_train, t_train = X[0:int(n*0.7)], y[0:int(n*0.7)], t[0:int(n*0.7)]
X_val, y_val, t_val = X[int(n*0.7):int(n*0.9)], y[int(n*0.7):int(n*0.9)], t[int(n*0.7):int(n*0.9)]
X_test, y_test, t_test = X[int(n*0.9):], y[int(n*0.9):], t[int(n*0.9):]

# ========== ML model.def ==========
model = Sequential()
model.add(InputLayer((10,2234)))
model.add(LSTM(64))
model.add(Dense(32, 'relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.3))
model.add(Dense(16, 'swish', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(32, 'relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(2233, 'linear'))
model.summary()

history_cb = tf.keras.callbacks.History()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=MeanSquaredError(),
              metrics=[RootMeanSquaredError()])
EPOCHS = 100
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    callbacks=[cp, history_cb, early_stopping])

print("Final Training Loss: ", history.history['loss'][-1], "Final Training RMSE: ", history.history['root_mean_squared_error'][-1])
print("Final Validation Loss:", history.history['val_loss'][-1], "Final Validation RMSE:", history.history['val_root_mean_squared_error'][-1], "\n")

model.save_weights('model.weights.h5')
model.load_weights('model.weights.h5')
predictions = model.predict(X_test)    # data format for printing:   X_train[0][:3][:1]   y_train[0][0]

print(X_test.shape)
print(predictions.shape)
print(y_test.shape)

predicted_values = scaler.inverse_transform(predictions) # This actually doesn't works as well as it should do
actual_values = scaler.inverse_transform(y_test)

print("var0, label: ", y_test[0][0], " predicted: ", predictions[0][0])
print("var34, label: ", y_test[0][34], " predicted: ", predictions[0][34])
print("var157, label: ", y_test[0][157], " predicted: ", predictions[0][157])
print("var1964, label: ", y_test[0][1964], " predicted: ", predictions[0][1964])
print("var2065, label: ", y_test[0][2065], " predicted: ", predictions[0][2065])

print("var0, label: ", actual_values[0][0], " predicted: ", predicted_values[0][0])
print("var34, label: ", actual_values[0][34], " predicted: ", predicted_values[0][34])
print("var157, label: ", actual_values[0][157], " predicted: ", predicted_values[0][157])
print("var1964, label: ", actual_values[0][1964], " predicted: ", predicted_values[0][1964])
print("var2065, label: ", actual_values[0][2065], " predicted: ", predicted_values[0][2065])

# ===== PLOT =====
plt.figure(figsize=(10,6))
plt.plot(t_test[0][:10], X_test[0][:10][:,1], label='val0')
plt.scatter(t_test[0][-1], y_test[0][0], label='label_val0')
plt.scatter(t_test[0][-1], predictions[0][0], label='predictions_val0')

plt.xlabel('Time')
plt.ylabel('val0')
plt.legend()
plt.grid(True)
plt.show()


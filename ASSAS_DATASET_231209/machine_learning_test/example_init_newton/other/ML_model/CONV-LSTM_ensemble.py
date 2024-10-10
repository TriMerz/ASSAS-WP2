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

ASTECROOT = "/opt/astecV3.1.1/"
COMPUTER  = "linux_64"
COMPILER  = "gccloc"
cache_file = 'data_cache.pkl'
bin_data = "mycesar_io.bin"
MinMaxWeights = 'min_max_scaler.pkl'

# ========== LOAD CACHED DATA or READ FROM *.bin SAVES ==========
if os.path.exists(cache_file):
   with open(cache_file, 'rb') as f:
      df = pickle.load(f)
   print("\n", "Cache loaded successfully!")
else:
   # ===== START ASTEC ENVIRONMENT =====
   sys.path.append(os.path.join(ASTECROOT, "code","proc"))
   sys.path.append(os.path.join(ASTECROOT, "code","bin", COMPUTER + "-" + "release", "wrap_python"))
   import AstecParser
   import astec
   AP = AstecParser.AstecParser()
   AP.parsed_arguments.compiler=COMPILER
   A = astec.Astec(AP)
   A.set_environment()
   import pyastec as pa
   pa.astec_init()

   varprim = []
   dtcesar = []
   dtmacro = []
   elapsed_time = []
   saved_database = len(os.listdir(bin_data))-2
   nb = 0

   print("\n")
   print("DATA EXTRACTION FROM: " + bin_data)
   for s,base in pa.tools.save_iterator(bin_data, t_start=None):
      if s != 0.0:
         nb +=1
         try:
            cardinality = len(list(base.family("CESAR_IO")))
            # == GET EVERY MACRO TIME-STEP ==
            v3 = base.get("CESAR_IO:dtmacro")
            dtmacro.append(v3)
            for i in range(cardinality):
               fam_name = "CESAR_IO " + str(i)
               # == GET ONLY CONVERGED SOLUTION ==
               conv = base.get(fam_name + ":CONV")
               if conv == 1:
                  # == GET VARPRIM ==
                  v1 = base.get(fam_name + ":OUTPUTS:VARPRIM")
                  varprim.append(v1)
                  # == GET dt ==
                  v2 = base.get(fam_name + ":dtfluid")
                  dtcesar.append(v2)
                  # == GET overall elapsed time ==
                  v4 = base.get(fam_name + ":STEPEND")
                  elapsed_time.append(v4)
               else:
                  pass
         except ModuleNotFoundError:
            print("CESAR_IO does not exist yet!")

         if (nb%10 == 0):
            progress = nb/saved_database * 100
            print(f"Reading... {progress:.2f}% complete")

   varprim = np.array(varprim)
   dtcesar = np.array(dtcesar)
   dtmacro = np.array(dtmacro)
   elapsed_time = np.array(elapsed_time)
   pa.end()
   # ===== BUILD THE DATAFRAME =====
   diff = []
   for j in range(len(varprim)):
      if j >= 1:
         v4 = varprim[j,:]-varprim[j-1,:]
         diff.append(v4)
      else:
         v4 = varprim[j,:]
   diff = np.array(diff)

   data = {'time': elapsed_time[1:], 'dtcesar': dtcesar[1:]} # !! the DataFrame begins with the second converged solution !!
   for i in range(diff.shape[1]):
      data[f'var{i}'] = diff[:, i]
   df = pd.DataFrame(data).sort_values(by='time')
   df.set_index('time', inplace=True)
   column_indices = {name: j for j, name in enumerate(df.columns)}
   # ===== NEW CASHED FILE =====
   with open(cache_file, 'wb') as f:
      pickle.dump(df, f)
   print("Cache file saved as:", cache_file)

# ========== NORMALIZATION ==========
features = df.drop(columns=['dtcesar'])

scaler = MinMaxScaler()
if os.path.exists(MinMaxWeights):
   scaler = joblib.load(MinMaxWeights)
   print("\n", "MinMaxScaler weights loaded successfully!")
else:
   scaler.fit(features.iloc[:int(len(df)*0.7),:])
   joblib.dump(scaler, MinMaxWeights)
norm_df = pd.DataFrame(scaler.transform(features), columns=features.columns, index=df.index)
norm_df.insert(0, 'dtcesar', df['dtcesar'])

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
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 10, 1, 2234, 1))
X_val, y_val, t_val = X[int(n*0.7):int(n*0.9)], y[int(n*0.7):int(n*0.9)], t[int(n*0.7):int(n*0.9)]
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 10, 1, 2234, 1))
X_test, y_test, t_test = X[int(n*0.9):], y[int(n*0.9):], t[int(n*0.9):]
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 10, 1, 2234, 1))

# ========== ML model.def ==========
model_convLSTM=Sequential()
model_convLSTM.add(InputLayer(input_shape=(10, 1, 2234, 1)))
model_convLSTM.add(ConvLSTM2D(64, (3, 3), activation='tanh', recurrent_activation='sigmoid',  return_sequences=True))
model_convLSTM.add(BatchNormalization())
model_convLSTM.add(Flatten())
model_convLSTM.add(LSTM(64, activation='relu', recurrent_activation='sigmoid'))
model_convLSTM.add(BatchNormalization())
model_convLSTM.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
model_convLSTM.add(Dropout(0.2))
model_convLSTM.add(Dense(16, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model_convLSTM.add(Dense(2233, activation='linear'))
model_convLSTM.summary()

model_LSTM=Sequential()
model_LSTM.add(InputLayer((10, 2234)))
model_LSTM.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid',  return_sequences=True))
model_LSTM.add(BatchNormalization())
model_LSTM.add(LSTM(64, activation='relu', recurrent_activation='sigmoid'))
model_LSTM.add(BatchNormalization())
model_LSTM.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
model_LSTM.add(Dropout(0.2))
model_LSTM.add(Dense(16, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model_LSTM.add(Dense(2233, activation='linear'))
model_LSTM.summary()

# == ensemble ==
merged_output = concatenate([model_convLSTM.output, model_LSTM.output])
ensemble_output = Lambda(lambda x: tf.keras.backend.mean(x, axis=1))(merged_output)
ensemble_model = tf.keras.Model(inputs=[model_convLSTM.input, model_LSTM.input], outputs=ensemble_output)

# cp = tf.keras.callbacks.ModelCheckpoint('model/ensemble_model.keras', save_best_only=True)
history_cb = tf.keras.callbacks.History()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
ensemble_model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss=MeanSquaredError(), metrics=['mae', 'mse'])
EPOCHS = 100
history = ensemble_model.fit([X_train_reshaped, X_train], y_train, validation_data=([X_val_reshaped, X_val], y_val), epochs=EPOCHS, callbacks=[history_cb, early_stopping])

print("Final Training Loss:", history.history['loss'][-1], "Final Training MSE:", (history.history['mse'][-1]) ** 0.5)
print("Final Validation Loss:", history.history['val_loss'][-1], "Final Validation MSE:", (history.history['val_mse'][-1]) ** 0.5, "\n")

ensemble_model.save_weights('ensemble_model.weights.h5')
ensemble_model.load_weights('ensemble_model.weights.h5')
predictions = ensemble_model.predict([X_train, X_train])    # data extraction for printing:   X_train[0][:3][:,1]   y_train[0][0]

print(X_test.shape)
print(predictions.shape)
print(y_test.shape)

predicted_values = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

# ===== PLOT =====
# plt.figure(figsize=(10,6))
# plt.plot(t_test[0][:10], X_test[0][:10][:,1], label='val0')
# plt.scatter(t_test[0][-1], y_test[0][0], label='label_val0')
# plt.scatter(t_test[0][-1], predictions[0][0], label='predictions_val0')

# plt.xlabel('Time')
# plt.ylabel('val0')
# plt.legend()
# plt.grid(True)
# plt.show()


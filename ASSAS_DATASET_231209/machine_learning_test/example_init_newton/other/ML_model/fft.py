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
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import *

# ===== Def. Fourier =====
class SpectralConv1D(tf.keras.layers.Layer):
    def __init__(self, out_channels, modes1):
        super(SpectralConv1D, self).__init__()

        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / out_channels)
        self.weights1 = tf.Variable(self.scale * tf.random.uniform([out_channels, modes1], dtype=tf.complex64(tf.float32)), trainable=True)

    def compl_mul1d(self, input, weights):
        return tf.einsum("bix,iox->box", input, weights)

    def call(self, x):
        batchsize = tf.shape(x)[0]
        x_ft = tf.signal.rfft(x)

        out_ft = tf.zeros([batchsize, self.out_channels, tf.shape(x)[-2] // 2 + 1], dtype=tf.complex64)
        out_ft[:, :, :self.modes1].assign(self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1))

        x = tf.signal.irfft(out_ft)
        return x

class MLP(tf.keras.Model):
    def __init__(self, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = layers.Conv1D(mid_channels, 1)
        self.mlp2 = layers.Conv1D(1, 1)

    def call(self, x):
        x = self.mlp1(x)
        x = tf.nn.gelu(x)
        x = self.mlp2(x)
        return x

class FNO1D(tf.keras.Model):
    def __init__(self, modes, width):
        super(FNO1D, self).__init__()

        self.modes1 = modes
        self.width = width

        self.p = layers.Dense(width)
        self.conv0 = SpectralConv1D(width, modes)
        self.conv1 = SpectralConv1D(width, modes)
        self.conv2 = SpectralConv1D(width, modes)
        self.conv3 = SpectralConv1D(width, modes)
        self.mlp0 = MLP(width)
        self.mlp1 = MLP(width)
        self.mlp2 = MLP(width)
        self.mlp3 = MLP(width)
        self.w0 = layers.Conv1D(width, 1)
        self.w1 = layers.Conv1D(width, 1)
        self.w2 = layers.Conv1D(width, 1)
        self.w3 = layers.Conv1D(width, 1)
        self.q = layers.Conv1D(2233, 1)

    def call(self, x):
        grid = tf.expand_dims(tf.linspace(0.0, 1.0, tf.shape(x)[1]), axis=0)
        grid = tf.tile(grid, [tf.shape(x)[0], 1, 1])
        x = tf.concat([x, grid], axis=-1)
        x = self.p(x)

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = tf.nn.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = tf.nn.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = tf.nn.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = self.q(x)
        return tf.transpose(x, perm=[0, 2, 1])

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
print("\n", "Windowed_DataBase.shape: ", X.shape, "\n", "Label.shape: ",y.shape)  # Windowed_DataBase.shape:  (99875, 10, 2234), Label.shape:  (99875, 2233)
print("Windowed_Time.shape: ", t.shape, "\n") # Windowed_Time.shape:  (99875, 11)

n = len(X)
X_train, y_train, t_train = X[0:int(n*0.7)], y[0:int(n*0.7)], t[0:int(n*0.7)]
X_val, y_val, t_val = X[int(n*0.7):int(n*0.9)], y[int(n*0.7):int(n*0.9)], t[int(n*0.7):int(n*0.9)]
X_test, y_test, t_test = X[int(n*0.9):], y[int(n*0.9):], t[int(n*0.9):]

# ========== ML model.def ==========
model_surrogate = Sequential()
model_surrogate.add(Input(shape=(WINDOW_SIZE, 2234)))
model_surrogate.add(Conv2D(filters=2233, kernel_size=(6, 6), activation='relu', padding='same'))
model_surrogate.add(MaxPool2D(pool_size=(1, 2233)))
model_surrogate.add(Conv2D(filters=4, kernel_size=(2, 2), activation='relu', padding='same'))
model_surrogate.add(Reshape((10, -1)))
model_surrogate.add(Conv1D(filters=WINDOW_SIZE, kernel_size=1, activation='relu'))
model_surrogate.add(LSTM(64, activation='relu', return_sequences=True))
model_surrogate.add(LSTM(64, activation='relu', recurrent_activation='sigmoid', return_sequences=True))
model_surrogate.add(LSTM(64, activation='relu', return_sequences=True))
model_surrogate.add(Flatten())
model_surrogate.add(Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model_surrogate.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
model_surrogate.add(Dropout(0.2))
model_surrogate.add(Dense(128, activation='relu'))
model_surrogate.add(Dropout(0.1))

# ===== Add Fourier Neural Operator 1D Layer =====
input_layer = Input(shape=(WINDOW_SIZE, 2234, 1))
fno_output = FNO1D(modes=16, width=64)(input_layer)
output_layer = Dense(2233, activation='linear')(fno_output)

model_with_fno = Model(inputs=input_layer, outputs=output_layer)
model_with_fno.summary()

history_cb = tf.keras.callbacks.History()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_with_fno.compile(optimizer=Adam(learning_rate=0.0001),
                       loss=MeanSquaredError(), metrics=['mae', RootMeanSquaredError()])
EPOCHS = 100
history = model_with_fno.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, callbacks=[history_cb, early_stopping])
print("Final Training Loss: ", history.history['loss'][-1], "Final Training RMSE: ", history.history['root_mean_squared_error'][-1])
print("Final Validation Loss:", history.history['val_loss'][-1], "Final Validation RMSE:", history.history['val_root_mean_squared_error'][-1], "\n")

model_with_fno.save_weights('2D1DConvlstm_model.weights.h5')
model_with_fno.load_weights('2D1DConvlstm_model.weights.h5')
predictions = model_with_fno.predict(X_test)    # data extraction for printing:   X_train[0][:3][:,1]   y_train[0][0]

print(X_test.shape)
print(predictions.shape)
print(y_test.shape)

predicted_values = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

print("var0, label: ", y_test[0][1], " predicted: ", predictions[0][1])
print("var34, label: ", y_test[0][34], " predicted: ", predictions[0][34])
print("var157, label: ", y_test[0][157], " predicted: ", predictions[0][157])
print("var1964, label: ", y_test[0][1964], " predicted: ", predictions[0][1964])
print("var2065, label: ", y_test[0][2065], " predicted: ", predictions[0][2065])

print("var0, label: ", actual_values[0][1], " predicted: ", predicted_values[0][1])
print("var34, label: ", actual_values[0][34], " predicted: ", predicted_values[0][34])
print("var157, label: ", actual_values[0][157], " predicted: ", predicted_values[0][157])
print("var1964, label: ", actual_values[0][1964], " predicted: ", predicted_values[0][1964])
print("var2065, label: ", actual_values[0][2065], " predicted: ", predicted_values[0][2065])

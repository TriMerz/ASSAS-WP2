#!/usr/bin/env python3

# ATTENTION all the packages must be imported also at the beginning of the file astec_main.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from IND import extract_index, outliers
import pyastec as pa

# I encountered some problems by importing the whole model instead of the weights only.
# For that reason at the beginning of the script I write the model, and then load the trained weights.
# == ML_1 model ==
model = Sequential()
model.add(InputLayer((10,2234)))
model.add(LSTM(64))
model.add(Dense(32, 'relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.3))
model.add(Dense(16, 'swish', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(32, 'relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(2233, 'linear'))
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
model.load_weights('model/model.weights.h5')

# The scaler's weights used during the training are loaded.
# == Scaler ==
MinMaxWeights = 'model/min_max_scaler.pkl'
scaler = MinMaxScaler()
if os.path.exists(MinMaxWeights):
  scaler = joblib.load(MinMaxWeights)
  # print("MinMaxScaler weights loaded...")
else:
  scaler.fit(features.iloc[:int(len(df)*0.7),:])
  joblib.dump(scaler, MinMaxWeights)

# == Some Functions ==
def df_to_X(df):
  X = np.expand_dims(df.to_numpy(), axis=0)
  return np.array(X)

class MyList(list):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._shape = (len(self),)
  @property
  def shape(self):
    return self._shape
  def update_list(self, t):
    self.pop(0)
    self.append(t)
    return self

"""-------------------------------------------------------
==========================================================
                  LOOP INITIALIZATION
==========================================================
-------------------------------------------------------"""
if "nb_pass" not in globals():
  print("Fisrt pass in this script...")
  nb_pass = 0
  w1 = MyList([None]*11)
  dtcesar = MyList([None]*10)
  time = MyList([None]*10)
  diff = MyList([None]*10)
  model.summary()
else:
  pass
print("""\
*-----------------------------------------------------------*
""",flush=True)
print("Pass number",nb_pass, "\n")

# Get all the needed data from ASTEC online calculation
# ===== DATA FROM ASTEC =====
root = pa.root_database()
ROOT = pa.PyOd_base(root)
newtoini = ROOT.get("CALC_OPT:CESAR:NEWTOINI")
varprim = newtoini.get("VARPRIM")
l = len(varprim)
w1.update_list(np.array(varprim))
v1 = ROOT.get("CESAR_IO:dtfluid")
dtcesar.update_list(v1)
v2 = ROOT.get("CESAR_IO:STEPEND")
time.update_list(v2)

# == LOOP ==
if(nb_pass >= 10):
  w2 = np.array(w1)
  for j in range(1, len(w2)):
    v4 = w2[j,:]-w2[j-1,:]
    diff.update_list(v4)
  valinc = np.array(diff)

  # The number of features is 2233, but to make the machine predict all the features given a time-step value, I added
  # the internal cesar time-step as first column.
  # == In-Loop DATAFRAME ==
  data = {'dtcesar': dtcesar}
  for i in range(valinc.shape[1]):
    data[f'var{i}'] = valinc[:, i]
  df = pd.DataFrame(data, index=None)

  # The normalization is made only to the 2233 features.
  # == FEATURES NORMALIZATION ==
  features = df.drop(columns=['dtcesar'])
  norm_df = pd.DataFrame(scaler.transform(features), columns=features.columns, index=df.index)
  norm_df.insert(0, 'dtcesar', df['dtcesar'])

  # update_list is used to update le list.
  # Combined with df_to_X is possible to have a window of the size (batch = 1, time_steps = 10, features = 2234)
  # == make a single batch (WINDOW) ==
  X = df_to_X(norm_df)
  print(X.shape)

  # == predictions ==
  predictions = model.predict(X)
  pred_valinc = np.squeeze(scaler.inverse_transform(predictions)).tolist()  # pa.PyOd_r1() use list object as input!! .lolist()
  pred = pa.PyOd_r1(pred_valinc)  # conversion to pa.PyOd_r1() object

  # At the beginning of the simulation, the amount of non-condensable gas is 0.0
  # The -- if x != 0.0 -- statement is used to update the features with the predicted incremental value, only if in the previous converged solution (i.e.) varprim[i] != 0
  # This is the first condition used to avoid ASTEC crash.
  # == new_varprim = varprim + diff (--> only if varprim[i] != 0.0)
  var = pa.PyOd_r1([x + y if x != 0.0 else x for x, y in zip(varprim, pred)]) # type(var): <class 'pyodessa.cls_vectors.PyOd_r1'>, len(var) = 2233

  # Impose the bopundary conditions
  indices = extract_index(var)
  var = outliers(var, indices)

  # update the varprim vector with the new values. Now this is the only way to reitrodue all the values inside ASTEC.
  for i in range(len(varprim)):
    e = var[i]
    varprim[i] = e

  # The following lines are not needed because by default "STAT" = 1, but sometimes forcing the reading process is suggested.
  # writing format: newtoini.put("variable_name", variable_value, overwrite = 0 / write = 1)
  # newtoini.put("STAT", 1, 0)
  # newtoini.put("NEWTOINI",varprim,0)

else:
  newtoini.put("STAT", 0, 0)

# == SHIFT nb_pass ==
nb_pass += 1

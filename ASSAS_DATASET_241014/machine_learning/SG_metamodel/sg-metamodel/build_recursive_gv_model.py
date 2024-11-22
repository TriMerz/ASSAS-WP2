#!/usr/bin/env python3

# favorite_queue = gpu
import pandas as pd
import numpy as np

import os
from os.path import join, isfile, isdir
import logging
import logging.config
import glob
from io import StringIO
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from datetime import datetime
import itertools

from tensorboard.plugins.hparams import api as hp

import gv_model as gvm
import gv_metamodel as gvtm
import joblib

FOLDER = 'sg_model'

FILE_PATTERN = FOLDER + "/*.pkl.gz"
#TODO limit to running dataset
#FILE_PATTERN = FOLDER + "/extracted-...is_run_0_reference_reduced.bin-gv4-1.pkl.gz"

raw_dataframes = dict(gvtm.generate_file_name_to_dataframe(FILE_PATTERN))
raw_dataframes = dict(list(gvtm.generate_file_name_to_dataframe(FILE_PATTERN))[0:4])

dataframes = {}

# Data smoother for v_liq, v_gas
for (file, raw_dataframe) in raw_dataframes.items():
    dataframes[file] = raw_dataframe
    
dataframe = pd.concat(dataframes.values())

# Data scaler
scaler = StandardScaler()
scaler.fit(dataframe)
normalized=scaler.transform(dataframe)
joblib.dump(scaler, os.path.join(FOLDER, 'scaler' ))

# Meta-model
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split


# Prepare data and split between train and test
METADATA_PATH = os.path.join(FOLDER, gvm.METADATA_FILE)
HORIZON_SIZE = 2

gv_model_metadata = gvm.load_gv_model_metadata(METADATA_PATH)

input_gather, output_gather, out_to_in_gather = gvtm.get_input_gather(gv_model_metadata), gvtm.get_output_gather(gv_model_metadata), gvtm.get_output_to_input_gather(gv_model_metadata)
output_index_to_name = gv_model_metadata.output_index_to_name

x_window_s = {}
y_s = {}

for (file, dataframe) in dataframes.items():
    normalized=scaler.transform(dataframe)
    ts = tf.convert_to_tensor(normalized)
    x_window, y = gvtm.normalized_to_x_window_and_y(ts,HORIZON_SIZE)
    x_window = gvtm.select_input_in_x_window(x_window, input_gather)
    y = gvtm.select_output_in_y(y, output_gather)
    x_window_s[file] = x_window
    y_s[file] = y

x_window:tf.Tensor = tf.concat(list(x_window_s.values()), axis=0)
y:tf.Tensor =  tf.concat(list(y_s.values()), axis=0)

x_train, x_test, y_train, y_test = train_test_split(x_window.numpy(), y.numpy(), shuffle=True, test_size=0.2)
x_train, x_test, y_train, y_test = tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test), 
x_train.shape, x_test.shape, y_train.shape, y_test.shape

LSTM_DIM = 400
DENSE_DIM_1 = 400
DENSE_DIM_2 = 350

sub_model = gvtm.GvModel(lstm_dim=LSTM_DIM, dense_dim_1=DENSE_DIM_1, dense_dim_2=DENSE_DIM_2)
sub_model.build((None, gvtm.WINDOW_SIZE, gvtm.INPUT_WIDTH))
model = gvtm.RecursiveGvModelExp( sub_model, out_to_in_gather, HORIZON_SIZE)
model.build((None, gvtm.WINDOW_SIZE + HORIZON_SIZE - 1, gvtm.INPUT_WIDTH))

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# Train model
EPOCHS = 100
BATCH_SIZE = 1024

datetime_now = datetime.now() 
datetime_now_str = f"{datetime_now.strftime('%Y%m%d-%H%M%S')}"

MODEL_PATH = join( FOLDER, "recursive_sg_metamodel" )

TRAIN = True

if TRAIN:
    
     
    history = model.fit(x_train,
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(x_test, y_test))
    
    sub_model.save(MODEL_PATH)


print(history.history.keys())
plt.plot(history.history["loss"],label='train loss')
plt.plot(history.history["val_loss"],label='test loss')

plt.title("Loss vs. Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)
plt.savefig('recursiveSgModelLoss.eps')

#plt.figure(1)
# plot(20)
if os.getenv( 'asbatch', None ) != '-batch' : plt.show()
ok=False

if( isdir( FOLDER ) ):
    if( isfile( join( FOLDER, 'metadata.pkl' ) ) ):
        ok = True

if(ok) :
    print ( "NORMAL END" )

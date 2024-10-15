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

raw_dataframes = dict(gvtm.generate_file_name_to_dataframe(FILE_PATTERN))
#raw_dataframes = dict(list(gvtm.generate_file_name_to_dataframe(FILE_PATTERN))[0:4])

ROLLING_MEAN_WINDOW = 10

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

gv_model_metadata = gvm.load_gv_model_metadata(METADATA_PATH)

input_gather, output_gather = gvtm.get_input_gather(gv_model_metadata), gvtm.get_output_gather(gv_model_metadata)
output_index_to_name = gv_model_metadata.output_index_to_name

x_window_s = {}
y_s = {}

for (file, dataframe) in dataframes.items():
    normalized=scaler.transform(dataframe)
    ts = tf.convert_to_tensor(normalized)
    x_window, y = gvtm.normalized_to_x_window_and_y(ts)
    y = tf.squeeze( y )
    x_window = gvtm.select_input_in_x_window(x_window, input_gather)
    y = gvtm.select_output_in_y(y, output_gather)
    x_window_s[file] = x_window
    y_s[file] = y

x_window:tf.Tensor = tf.concat(list(x_window_s.values()), axis=0)
y:tf.Tensor =  tf.concat(list(y_s.values()), axis=0)

x_train, x_test, y_train, y_test = train_test_split(x_window.numpy(), y.numpy(), shuffle=True, test_size=0.2)
x_train, x_test, y_train, y_test = tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test), 
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# Build model
def get_capture_print():
    content = []
    def capture_print(new_content:str):
        content.append(new_content)
    return capture_print, content

LSTM_DIM = 400
DENSE_DIM_1 = 400
DENSE_DIM_2 = 350

model = gvtm.GvModel(lstm_dim=LSTM_DIM, dense_dim_1=DENSE_DIM_1, dense_dim_2=DENSE_DIM_2)
model.build((None, gvtm.WINDOW_SIZE, gvtm.INPUT_WIDTH))

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

CHECKPOINT_DIR = ".tf.checkpoints"
TENSORBOARD_DIR = ".tf.tensorboard"
CHECKPOINT_PATH = CHECKPOINT_DIR + "/gv_model-{epoch:04d}.ckpt"
TENSORBOARD_LOG_DIR = f".tf.tensorboard/{datetime_now_str}"
TENSORBOARD_BASELINE_LOG_DIR = TENSORBOARD_DIR + f"/baseline-{datetime_now_str}"
MODEL_PATH = join( FOLDER, "sg_metamodel" )

TRAIN = True

if TRAIN:
    
    tensorboard = keras.callbacks.TensorBoard(log_dir = TENSORBOARD_LOG_DIR,
                                          histogram_freq = 1)
    
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                             save_weights_only=True,
                                             save_freq='epoch',
                                             verbose=0)
    
    file_writer = tf.summary.create_file_writer(TENSORBOARD_LOG_DIR)
    
    with file_writer.as_default():
        capture_print, capture_content = get_capture_print()
        summary_content = model.summary(print_fn=capture_print)
        capture_content_str = "  \n".join(capture_content)
        tf.summary.text("model", capture_content_str, step=0)
    
    history = model.fit(x_train,
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard, checkpoint])
    
    model.save(MODEL_PATH)
else:
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    print("Latest checkpoint : ", latest_checkpoint)
    model.load_weights(latest_checkpoint)
    
    model.save_weights(MODEL_PATH)

## Worst variables
## EVALUATION_STEPS = 10

## history = model.evaluate(
##     x_test,
##     y_test,
##     batch_size=BATCH_SIZE,
##     steps=EVALUATION_STEPS,
##     return_dict=True
## )

## history_without_loss = history.copy()
## history_without_loss.pop("loss")

## def plot(max:int):
##     sorted_history = {k: v for k, v in sorted(history_without_loss.items(), key=lambda item: item[1], reverse=True)}
##     x = []
##     y = []
##     for (name, value) in itertools.islice(sorted_history.items(), max):
##         x.append(name)
##         y.append(value)
##     plt.bar(x=x, height=y)
##     plt.xticks(rotation='vertical')
##     plt.title(f"Absolute error on major {max} output variable")
##     if os.getenv( 'asbatch', None ) != '-batch' : plt.show()

print(history.history.keys())
plt.plot(history.history["loss"],label='train loss')
plt.plot(history.history["val_loss"],label='test loss')

plt.title("Loss vs. Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)
plt.savefig('sgmodelLoss.eps')

#plt.figure(1)
# plot(20)
if os.getenv( 'asbatch', None ) != '-batch' : plt.show()
ok=False

if( isdir( FOLDER ) ):
    if( isfile( join( FOLDER, 'metadata.pkl' ) ) ):
        ok = True

if(ok) :
    print ( "NORMAL END" )

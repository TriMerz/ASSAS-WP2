#!/usr/bin/env python3
import os
from os.path import join, isfile, isdir
import gv_model as gvm
import gv_metamodel as gvtm
import keras.models
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import joblib
from numpy import linalg as LA


DIR_PATH = "sg_model"

FILE_PATTERN = join( DIR_PATH, "*.pkl.gz" )
METADATA_PATH = join( DIR_PATH, "metadata.pkl" )

gv_model_metadata = gvm.load_gv_model_metadata(METADATA_PATH)

scaler = joblib.load(join(DIR_PATH, 'scaler' ))

MODEL_PATH = join( DIR_PATH, "sg_metamodel" )

model = keras.models.load_model(MODEL_PATH)
model.summary()

def update_window(old_window:tf.Tensor, x:tf.Tensor, y_pred:tf.Tensor) -> tf.Tensor:
    window = old_window[1:] # Forget oldest input

    editable_x = x.numpy()
    for (output_index, input_index) in gv_model_metadata.output_index_to_input_index.items():
        editable_x[input_index] = y_pred[output_index]
    
    edited_x = tf.convert_to_tensor(editable_x)
    edited_x = tf.expand_dims(edited_x, axis=0)
    result = tf.concat([window, edited_x], axis=0)
    return result

def prediction(xs_window:tf.Tensor, model:gvtm.GvModel) -> tf.Tensor:
    time_steps:int = xs_window.shape[0]
    assert xs_window.shape[1] == gvtm.WINDOW_SIZE
    assert xs_window.shape[2] == gvtm.INPUT_WIDTH
    
    outputs = []

    for index in tqdm(range(time_steps),'Prediction phase'):
        input = xs_window[index]
        batch_input = tf.expand_dims(input, axis=0)

        batch_output = model.predict(batch_input, verbose=0)

        output = batch_output[0]
        
        outputs.append(output)

    return tf.convert_to_tensor(outputs)

def simulation(xs_window:tf.Tensor, model:gvtm.GvModel) -> tf.Tensor:
    time_steps:int = xs_window.shape[0]
    assert xs_window.shape[1] == gvtm.WINDOW_SIZE
    assert xs_window.shape[2] == gvtm.INPUT_WIDTH
    
    outputs = []
    reinit = 0
    for index in tqdm(range(time_steps),'Simulation phase'):
        if reinit % 200 == 0:
            input = xs_window[index]
        reinit = reinit + 1
        batch_input = tf.expand_dims(input, axis=0)

        batch_output = model.predict(batch_input, batch_size=1, verbose=0)

        output = batch_output[0]
        
        outputs.append(output)
        if index < time_steps-1 :
            input = update_window(input, xs_window[index+1][-1], output)

    return tf.convert_to_tensor(outputs)

file_names = list(gvtm.generate_file_names(FILE_PATTERN))

# Use first batch
file_name = file_names[7]
input_gather = gvtm.get_input_gather(gv_model_metadata)
output_gather = gvtm.get_output_gather(gv_model_metadata)

dataframe = gvtm.file_name_to_dataframe(file_name)
ROLLING_MEAN_WINDOW=10
dataframe_copy = dataframe.copy()
v_s_cols = [col for col in dataframe.columns if ("v_liq" in col) or ("v_gas" in col)]
dataframe_copy[v_s_cols] = dataframe[v_s_cols].rolling(ROLLING_MEAN_WINDOW, center=True, min_periods=1).mean()
dataframe = dataframe_copy
normalized_dataframe = scaler.transform(dataframe)
normalized = tf.convert_to_tensor(normalized_dataframe)
xs_window_full, ys_full_norm = gvtm.normalized_to_x_window_and_y(normalized)
ys_full_norm = tf.squeeze( ys_full_norm )
xs_window = gvtm.select_input_in_x_window(xs_window_full, input_gather)
ys_full=scaler.inverse_transform( ys_full_norm )
ys = gvtm.select_output_in_y(ys_full, output_gather)

simu = simulation(xs_window, model)
pred = prediction(xs_window, model)
ys_norm = gvtm.select_output_in_y(ys_full_norm, output_gather)

diff=ys_norm.numpy()-simu.numpy()
diff_L2=LA.norm(diff, axis=0)
perm=diff_L2.argsort()

simulation_result = gvtm.scale_sub_tensor( simu, output_gather, scaler, True )
prediction_result = gvtm.scale_sub_tensor( pred, output_gather, scaler, True )

figure_index=0
ext='.jpg'
n = ys.shape[1]
for i in range(20):
    j=perm[len(perm)-1-i]
    idx=output_gather[j].numpy()
    lab=dataframe.columns[idx]
    print ( lab + ' norm ' + str( diff_L2[j] ) )
    if True: #'CAVX' in lab: 
            plt.figure(figure_index)
            ys_true = ys[:,j].numpy()
            ys_pred = prediction_result[:, j].numpy()
            ys_simu = simulation_result[:, j].numpy()
            ts = dataframe.index[gvtm.WINDOW_SIZE:].array
            plt.plot(ts,ys_true,'-or',label='ASTEC ')
            plt.plot(ts,ys_pred,'b',label='pred ')
            plt.plot(ts,ys_simu,'g',label='simu ')
            plt.title(lab)
            plt.xlabel("Iteration")
            plt.ylabel(lab)
            plt.legend()
            figname='compare_'+str(figure_index)
            plt.savefig(figname+ext)
            
    figure_index=figure_index+1
if os.getenv( 'asbatch', None ) != '-batch' : plt.show()
print ( "NORMAL END" )

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

import glob
import logging
import pandas as pd
from typing import Tuple
import gv_model as gvm

WINDOW_SIZE = 5
INPUT_WIDTH = 318
OUTPUT_WIDTH = 238

def generate_file_names(file_pattern:str):
    return glob.glob(file_pattern)

def _convert_object_to_catagorical(df: pd.DataFrame):
    def categorical_to_index(column:pd.Series):
        return column.cat.codes
    object_columns = df.columns[df.dtypes == object]
    df[object_columns] = df[object_columns].astype('category').apply(categorical_to_index)


def file_name_to_dataframe(file_path:str):
    logging.info(f"Reading pickle file {file_path}")
    df = pd.read_pickle(file_path, compression="gzip")

    _convert_object_to_catagorical(df)

    return df

def generate_file_name_to_dataframe(file_pattern:str):
    for file_name in generate_file_names(file_pattern):
        yield file_name, file_name_to_dataframe(file_name)

def normalize_dataframe(dataframe:pd.DataFrame):
    mean, std = dataframe.mean(), dataframe.std()

    std = std.replace(0, 1)

    return (dataframe - mean) / std

def normalized_to_x_window_and_y(normalized:tf.Tensor, window_size:int=WINDOW_SIZE, horizon_size:int=1):
    ys_s = []
    ns = normalized.shape[0]
    for index in range(horizon_size):
        ys_s.append(normalized[window_size + index : ns + 1 -horizon_size + index  , :])
    ys = tf.stack(ys_s)
    y = tf.transpose(ys, perm=[1, 0, 2])
    ts_s = []
    for index in range(window_size+horizon_size-1):
        ts_s.append(normalized[index:-window_size + index - horizon_size + 1,:])
    x = tf.stack(ts_s)
    x = tf.transpose(x, perm=[1, 0, 2])
    return x, y

def scale_sub_tensor( y:tf.Tensor, index_gather:tf.Tensor, scaler, inverse=False ) -> tf.Tensor:

    tmp = np.zeros([y.shape[0],len(scaler.get_feature_names_out())])
    for i in range(  index_gather.shape[0] ):
        tmp[:,index_gather[i]] = y[:,i]
    if inverse:
        tmp = scaler.inverse_transform( tmp )
    else:
        tmp = scaler.transform( tmp )
           
    return tf.gather(tmp, index_gather, axis=1)

def select_input_in_x_window(x_window:tf.Tensor, input_gather:tf.Tensor):
    # assert input_gather.shape[0] == INPUT_WIDTH
    return tf.gather(x_window, input_gather, axis=2)

def select_output_in_y(y:tf.Tensor, output_gather:tf.Tensor):
    # assert output_gather.shape[0] == OUTPUT_WIDTH
    return tf.gather(y, output_gather, axis=len(y.shape)-1)

def scatter_output_in_full_y(y_s:tf.Tensor, output_gather:tf.Tensor):
    # assert output_gather.shape[0] == OUTPUT_WIDTH
    return tf.scatter_nd(indices=output_gather, updates=y_s, shape=[ y_s.shape[0], INPUT_WIDTH ] )

def get_input_gather(gv_model_metadata:gvm.GvModelMetadata):
    return tf.convert_to_tensor(tuple(gv_model_metadata.input_indexes))

def get_output_gather(gv_model_metadata:gvm.GvModelMetadata):
    return tf.convert_to_tensor(tuple(gv_model_metadata.output_indexes))

def get_output_to_input_gather(gv_model_metadata:gvm.GvModelMetadata):
    return tf.convert_to_tensor(tuple(gv_model_metadata.output_index_to_input_index))

class GvBaseline(keras.Sequential):

    def __init__(self, output_dim:int=OUTPUT_WIDTH):
        super().__init__([
            layers.Flatten(),
            layers.Dense(output_dim, name="dense_output")
        ], name="gv_baseline")

class GvModel(keras.Sequential):

    def __init__(self, lstm_dim:int, dense_dim_1:int, dense_dim_2:int, output_dim:int=OUTPUT_WIDTH):
        super().__init__((
            layers.LSTM(lstm_dim, name="lstm"),
            layers.Dense(dense_dim_1, name="dense_1", activation="relu"),
            layers.Dense(dense_dim_2, name="dense_2", activation="relu"),
            layers.Dense(output_dim, name="dense_3")
        ), name="gv_model")

class RecursiveGvModelExp(keras.Model):

    def __init__(self, sub_model:keras.Model, indices, horizon_size:int ):
        super(RecursiveGvModelExp, self).__init__(name="recursive_gv_exp_model")
        self.submodel = sub_model
        self.horizon_size = horizon_size
        nindices = indices.numpy()
        list=[]
        for i,j in enumerate ( indices.numpy() ):
            list.append( [ j , WINDOW_SIZE-1 ] )
        self.indices = tf.convert_to_tensor(list)

    def call(self, inputs, training=None, mask=None):

        outputs = []
##         bs = tf.shape(inputs)[0]
##         tf.print('Input shape ')
##         tf.print(tf.shape(inputs))
        for horizon in range(self.horizon_size):
            window = inputs[:,horizon:WINDOW_SIZE+horizon,:]
            if( horizon > 0 ) :
##                 tf.print('Horizon '+str(horizon))
##                 tf.print('Input window 0 ')
##                 tf.print(window[0,:,0])
##                 tf.print('Output ')
##                 tf.print(tf.shape(output))
##                 tf.print(output[:,0])
                new_window = tf.tensor_scatter_nd_update( tf.transpose( window ), self.indices, tf.transpose( output) )
                window = tf.transpose( new_window )               
##                 tf.print('Output window 0 ')
##                 tf.print(window[0,:,0])
            output = self.submodel.call(window,training,mask)
            outputs.append( output )
## ##             print(output.shape)
##             print(tf.expand_dims(self.indices, 0).shape)
##             print((bs,inputs.shape[2]))
##             print(self.indices)
##             scattered_output = tf.scatter_nd( tf.expand_dims(self.indices, 1), output, shape = [ bs, inputs.shape[2] ] )
##             print(scattered_output.shape)
##            scattered_output = tf.scatter_nd( self.indices, tf.transpose( output) , shape = [ 1000, WINDOW_SIZE, bs ] )
##            print(f"scattered shape {scattered_output.shape}")

        result = tf.stack(outputs)
        result = tf.transpose(result, perm=[1, 0, 2])
        return result

class SliceLayer(layers.Layer):

    def __init__(self, slice:Tuple, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.slice = slice
    
    def call(self, inputs, *args, **kwargs):
        return inputs[self.slice]

class GvBiModel(keras.Model):

    def __init__(self, lstm_dim:int, dense_dim_1:int, dense_dim_2:int, output_dim:int=OUTPUT_WIDTH):
        super(GvBiModel, self).__init__(name="gv_bi_model")
        self.lstm = layers.LSTM(lstm_dim, name="lstm")
        self.last_in_window = SliceLayer((slice(None, None, 1),-1,slice(None, None, 1)),name="last_in_window")
        self.concat = layers.Concatenate(axis=1)
        self.dense_1 = layers.Dense(dense_dim_1, name="dense_1", activation="relu")
        self.dense_2 = layers.Dense(dense_dim_2, name="dense_2", activation="relu")
        self.dense_output = layers.Dense(output_dim, name="dense_output")

    def call(self, inputs, training=None, mask=None):
        lstmed = self.lstm(inputs)
        last_input = self.last_in_window(inputs)
        concatenated = self.concat([last_input, lstmed])
        densed_1 = self.dense_1(concatenated)
        densed_2 = self.dense_2(densed_1)
        return self.dense_output(densed_2)

class WindowAdapter:

    def __init__(self, window_size:int, input_width:int) -> None:
        self.window_size = window_size
        self.input_width = input_width
        self.arrays = []

    def add_input(self, input:np.ndarray):
        assert input.shape[0] == self.input_width

        if len(self.arrays) == 0:
            self.arrays = [input for _ in range(self.window_size)]
        elif len(self.arrays) == self.window_size:
            self.arrays = self.arrays[-self.window_size + 1:] + [input]
        else:
            raise Exception(f"Invalid window size : {len(self.arrays)}, expected 0 or {self.window_size}")
    
    def get_window(self):
        assert len(self.arrays) == self.window_size

        result = np.array(self.arrays)

        assert result.shape == (self.window_size, self.input_width)

        return result


def metamodel_step(model:keras.Model, inputs:np.ndarray, gv_model_metadata:gvm.GvModelMetadata, scaler ) -> np.ndarray:
    assert inputs.shape[0] == WINDOW_SIZE
    assert inputs.shape[1] == INPUT_WIDTH

    input_gather = get_input_gather(gv_model_metadata)
    output_gather = get_output_gather(gv_model_metadata)

    tensor:tf.Tensor = tf.convert_to_tensor(inputs)
    scaled_batch = scale_sub_tensor( tensor, input_gather, scaler )
    batch = tf.expand_dims(scaled_batch, axis=0)
    output_batch = model(batch)
    output_unscaled = scale_sub_tensor( output_batch, output_gather, scaler, True )
    output:tf.Tensor = tf.squeeze(output_unscaled, axis=0)
   
    return output.numpy()

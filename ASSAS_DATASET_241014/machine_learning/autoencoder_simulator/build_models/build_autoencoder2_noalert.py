#!/usr/bin/env python3
# favorite_queue = gpu
import os.path
import sys

import pandas
import numpy
import astec
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import linalg as LA
import joblib
from common import die

def create_model(N,reduced_N):

    x = inputs = keras.Input(shape=(N,), name="img")
    dims=[N]
    activation=[None]
    n=int(N/2)
    while(n>reduced_N):
        dims.append(n)
        activation.append('selu')
        n=int(n/2)
    dims.append(reduced_N)
    activation.append(None)

    for i in range(len(dims)):
        x = layers.Dense(dims[i],activation=activation[i])(x)
    outputs = x
    encoder = keras.Model(inputs=inputs, outputs=outputs, name="encoder")
    encoder.summary()

    decoder_input = keras.Input(shape=(reduced_N,), name="encoded_img")
    x=decoder_input
    #activation=[None,'relu','selu','selu','selu','selu','selu','selu']
    for i in range(len(dims)):
        x = layers.Dense(dims[len(dims)-1-i],activation=activation[len(dims)-1-i])(x)

    decoder_output = x

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")
    decoder.summary()

    autoencoder_input = keras.Input(shape=(N,), name="img")
    encoded_img = encoder(autoencoder_input)
    decoded_img = decoder(encoded_img)
    autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
    autoencoder.summary()

    autoencoder.compile(
        loss='mse',#keras.losses.MeanSquaredError(),
        optimizer='Adam', #keras.optimizers.RMSprop(),
        metrics=["accuracy"]
    )

    return autoencoder

######################################################
# Start
#######

#####################################################
# Read scenarios
###################################################
modeltype="autoencoder"
ext='.eps'
all_runs="all_runs_circuit.csv.zip"

#
first_sample = joblib.load('first_sample.sav')
print('first_sample '+str(first_sample))
nbrun=len(first_sample)-1

print('Read '+all_runs)
df=pandas.read_csv(all_runs,sep=';')
#Remove index
df = df.drop('time',axis=1)

df_is_null=df.isnull()
if df_is_null.values.any():
    for l in df_is_null:
        if df_is_null[l].any():
            print(df[l])
            print( "Column "+l+" contains null value" )

    print( "A null value exists" )
    df = df.fillna( 0. )

labels=df.columns
joblib.dump(labels, modeltype+'labels.sav')

#

df.info()

N=len(labels)
print('Sample dimension: '+str(df.shape[0]))
print('Number of simulations: '+str(nbrun))
print('Number of features: '+str(N))
df=df.to_numpy()

###################################################################
# Separate between training data and validation data
print('Split between train and test')
train, test, y_train, y_test = train_test_split(df,df,random_state=20)
#train = df.iloc[0:first_sample[nbrun-1]]
##################################################################
# Build and apply scaling
print('Scale sampling wrt train')
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)
df_scal = scaler.transform(df)

# Global var
figure_index=0

###################################################################
# Create and fit or restore model
#########################

model_filename="autoencoder_model"
rebuild=not os.path.exists(model_filename)

# Latent space dimension
reduced_N=10

if(rebuild):
    # Create and fit
    print('Build and train '+modeltype)
    autoencoder=create_model(N,reduced_N)
    history = autoencoder.fit(train, train, batch_size=10*N, epochs=150)
    autoencoder.save(model_filename)
    plt.figure(0)
    plt.plot(history.history["loss"],label='train loss')

    plt.title("Loss vs. Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig('autoencoderLoss'+ext)
    figure_index=figure_index+1

else:
    print('Restore '+modeltype)
    # Restore
    autoencoder = keras.models.load_model(model_filename)

joblib.dump(scaler, modeltype+'_scaler')

encoder=autoencoder.layers[-2]

# Evaluate prediction on all samples and L2 norm prediction

df_pred=autoencoder.predict(df_scal)
df_pred_real=scaler.inverse_transform(df_pred)

df_diff=df_pred-df_scal
df_diff_L2=LA.norm(df_diff, axis=0)
perm=df_diff_L2.argsort()

print("Score on total sample: ", autoencoder.evaluate(df_scal, df_scal, verbose=0) )
print("Score on validation sample: ", autoencoder.evaluate(test, test, verbose=0) )

# Display some curves

#for k in range(10):
for k in range(0,N,100):
    i=perm[N-k-1]
    i=k
    lab=labels[i]+' err. '+str(df_diff_L2[i])
    lab=labels[i]
    plt.figure(figure_index,figsize=(12,8))
    for c in range(0,nbrun,2):
        col=plt.cm.RdYlBu(c*1.0/nbrun)
        plt.plot(df[first_sample[c]:first_sample[c+1],i],label='Run '+str(c),color=col)
        plt.plot(df_pred_real[first_sample[c]:first_sample[c+1],i],label='Predicted '+str(c),color=col,linestyle='dashed')
    plt.title(lab)
    plt.xlabel("Iteration")
    plt.ylabel(lab)
    plt.legend()
#    figname=labels[i].replace("'","").replace(" ","").replace("(","_").replace(")","_").replace("[","_").replace("]","_")
    figname='autoencoded'+str(figure_index)
    plt.savefig(figname+ext)

    figure_index=figure_index+1

######################################################################
# Build and save latent spaces
##############################

latent=encoder.predict(df_scal)

for i in range(nbrun):
    dataframe = pandas.DataFrame(latent[first_sample[i]:first_sample[i+1]], dtype = float)
    filename="latent_run"+str(i)+".csv.zip"
    dataframe.to_csv(filename,sep=';')

if os.getenv( 'asbatch', None ) != '-batch' : plt.show()
print ( "NORMAL END" )

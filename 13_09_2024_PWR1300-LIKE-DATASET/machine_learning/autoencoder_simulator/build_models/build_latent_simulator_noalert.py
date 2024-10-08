#!/usr/bin/env python3
# favorite_queue = gpu
import os.path
import sys
import pandas
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from numpy import linalg as LA

# split a univariate sequence into samples
def split_sequence(sequence, n_steps, n_pred):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# find the end of this pattern
		end_iy = i + n_steps + n_pred
		# check if we are beyond the sequence
		if end_iy > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_iy]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def create_model(n_steps, n_features, n_pred):
        model = Sequential()
##        model.add(LSTM(50, input_shape=(n_steps, n_in)))
        model.add(LSTM(100, input_shape=(n_steps, n_features), return_sequences=True))
        model.add(LSTM(100, input_shape=(n_steps, n_features)))
##        model.add(Dense(4*n_features*n_pred,activation='relu'))
##        model.add(Dense(2*n_features*n_pred,activation='relu'))
##        model.add(Dense(10*n_features*n_pred,activation='relu'))
        model.add(Dense(n_features*n_pred))
        model.compile(optimizer='RMSprop', loss='mse', metrics=["accuracy"])

        model.summary()

        return model


def read_samples(file_base):
    first_sample=[0]
    i=0
    while(True):
        filename=file_base+str(i)+".csv.zip"
        if os.path.exists(filename):
                print("Reading "+filename)
                dfi = pandas.read_csv(filename,sep=';')
                if(i==0):
                        df=dfi
                else: 
                        df=pandas.concat([df,dfi],axis=0)
                first_sample.append(df.shape[0])
                i=i+1
        else:
                break

    return [ df, first_sample ]

n_steps=10
n_pred=n_steps
figure_index=0

modeltype='latentSimulator'
all_runs_filename=modeltype+"_all_runs.csv.zip"
#
if os.path.exists(all_runs_filename):
   df=pandas.read_csv(all_runs_filename,sep=';')
   df=df.iloc[:,1:]
   first_sample = joblib.load(modeltype+'first_sample.sav')
#Remove index
else:
   print('Read latent_run*')
   df, first_sample = read_samples("latent_run")
#Remove index
   df=df.iloc[:,1:]
   df.to_csv(all_runs_filename,sep=';',index_label='time')
   joblib.dump(first_sample, modeltype+'first_sample.sav')

nbrun=len(first_sample)-1

# Labels of the dataset
labels=df.columns

n_features=len(labels)
n=df.shape[0]
   
# Scaler should be trained only on train set but here, its more convenient
print("Scale sampling")
X_scaler = StandardScaler()
print(df.shape)
X_scaler.fit(df)
sequence = X_scaler.transform(df)

# Transform time sequence in new variables then split between train and test
new_first=[0]
print('Sampling shape '+str(sequence.shape))
print("Build evolutive sampling matrix-vector")
for irun in range(nbrun):
        print("  Add simulation "+str(irun))
        istart=first_sample[irun]
        iend=first_sample[irun+1]
        Xi,yi = split_sequence(sequence[istart:iend,:],n_steps,n_pred)
        if(irun==0):
                X=Xi
                y=yi
        else:
                X=np.concatenate((X,Xi),axis=0)
                y=np.concatenate((y,yi),axis=0)
        new_first.append(X.shape[0])

first_sample=new_first

print("Split between train and test")
X_train, X_test, y_train, y_test = train_test_split(X[first_sample[0]:first_sample[nbrun-1]], y[first_sample[0]:first_sample[nbrun-1]], test_size=0.1,random_state=20)

nv=X_train.shape[0]

# define model
X_train = X_train.reshape((nv, n_steps, n_features))
X_test = X_test.reshape((X_test.shape[0], n_steps, n_features))
y_train = y_train.reshape((nv,n_pred*n_features))

###################################################################
# Create and fit or restore model
#########################

model_filename=modeltype+"_model"
rebuild=not os.path.exists(model_filename)
joblib.dump(X_scaler, modeltype+'_scaler')
ext='.eps'

if(rebuild):
    print('Build an train simulator')
    # Create and fit
    simulator=create_model(n_steps, n_features, n_pred)
    history = simulator.fit(X_train, y_train, epochs=10, verbose=2)
    simulator.save(model_filename)
    plt.figure(0)
    plt.plot(history.history["loss"],label='train loss')
    
    plt.title("Loss vs. Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig('latentSimulatorLoss'+ext)
    figure_index=figure_index+1
    
else:
    print('Read simulator')
    # Restore
    simulator = keras.models.load_model(model_filename)

# demonstrate prediction
nv=X_test.shape[0]
y_test = y_test.reshape((nv,n_pred*n_features))
test_scores = simulator.evaluate(X_test,y_test, verbose=2)
y_test = y_test.reshape((nv,n_pred,n_features))
print("Accuracy on test: ", test_scores)
Y=simulator.predict(X_test, verbose=0)
Y = Y.reshape((nv,n_pred,n_features))
print("Max error "+str((Y-y_test).max()))
df_diff_L2=LA.norm(Y-y_test, axis=0)
print("L2 norm: ")
print(df_diff_L2)

# Simulation
nv=X.shape[0]
y_simu_tot=np.zeros((nv,n_features))
x_simu=X[0:1,:,:]
nb_resets=10*n_pred
to_show=[0,int(nbrun/2),nbrun-1]
for irun in to_show:
    print('Simulates run '+str(irun))
    istart=first_sample[irun]
    iend=first_sample[irun+1]
    idx=0
    for k in range(istart,iend-n_pred-1,n_pred):
            if(idx%nb_resets==0):
                    print('Reset solution at iteration '+str(k))
                    x_simu=X[k:k+1,:,:].copy()
            y_simu_tot[k:k+n_pred,:]=x_simu[0,:,:].copy()
            idx=idx+n_pred
            y_simu=simulator.predict(x_simu)
            y_simu=y_simu.reshape(1,n_pred,n_features)
            x_simu[0,:,:]=y_simu[0,:,:].copy()

yd=y[:,0,:]
Y=simulator.predict(X, verbose=0)
Y = Y.reshape(Y.shape[0],n_pred,n_features)
yp=Y[:,0,:]
ys=y_simu_tot
for i in range(n_features):
    plt.figure(figure_index)
    # Plot first run and valid one
    for j in to_show:
            #plt.plot(sequence[first_sample[j]+n_pred:first_sample[j+1],i],linewidth=4,label='Run X '+str(j))
            plt.plot(yd[first_sample[j]:first_sample[j+1],i],'o',label='Run y'+str(j))
            plt.plot(yp[first_sample[j]:first_sample[j+1],i],linewidth=4,label='Predicted '+str(j))
            plt.plot(ys[first_sample[j]+n_pred:first_sample[j+1],i],linewidth=4,label='Simulated '+str(j))
    plt.title('Latent variable '+str(i)+' reset every '+str(nb_resets))
    plt.xlabel("Iteration")
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('latentVariable'+str(i)+ext)
    figure_index=figure_index+1


if os.getenv( 'asbatch', None ) != '-batch' : plt.show()
print ( "NORMAL END" )

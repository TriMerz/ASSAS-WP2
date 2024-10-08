#!/usr/bin/env python3
import os
import os.path
import sys
import pandas
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import linalg as LA
import joblib
from common import die

n_steps=10
latent_dim=10

# This script is dedicated to illustrating difference between real solution and predicted or only encoded_decoded solution
#

class Test:

    def __init__(self):
        self.logfile = open('log.lst','w')
        self.read_models()
        
    def model_filename(self,model):
        models_path="."
        return os.path.join(models_path,model)
    
    def log(self, msg):
      self.logfile.write(msg+"\n")
      self.logfile.flush()
      
    def read_models(self):

        self.log("Load encoder-decoder")
        self.autoencoder = keras.models.load_model(self.model_filename("autoencoder_model"))
        self.encoder=self.autoencoder.layers[-2]
        self.decoder=self.autoencoder.layers[-1]
        self.encoder_scaler = joblib.load(self.model_filename('autoencoder_scaler'))

        self.log("Load path of the variables")
        self.labels = joblib.load(self.model_filename('autoencoderlabels.sav'))
        self.current_variables = np.zeros((1,len(self.labels)))

        self.log("Load latent simulator")
        self.simulator = keras.models.load_model(self.model_filename("latentSimulator_model"))
        self.simulator_scaler = joblib.load(self.model_filename('latentSimulator_scaler'))
        self.x_simu = np.zeros((1,n_steps,latent_dim))

    # Compare fiducial solution with simulated one for npred*n_steps iterations without restarting
    # starting at iter iteration
    def compute_simulated(self,npred):
        nb_samples=self.real_variables.shape[0]
        
        # For each prediction, we need n_steps fiducial solution and we will predict npred*n_steps iteration
        # so each cycle concerns (npred+1)*n_steps iterations
        shift=(npred+1)*n_steps
        nb_cycle=int( nb_samples/ shift )
        nb_iterations=  nb_cycle*shift          

        print( "Compute predicted solution considering "+str(nb_cycle)+" cycles with "+str(npred)+" predictions")
        # We initialize simulated variable with real variable to fill iterations not simulated
        self.simu_variables = self.real_variables.copy()
        
        # Prepare real_variables for simulation
        scaled_encoded_variables = self.simulator_scaler.transform( self.encoded_variables )

        # Predict from real_variables
        for icycle in range(nb_cycle):
            start_iteration=icycle*shift
            # Prepare x_simu receiving true solution
            for i in range(n_steps):
                for j in range(latent_dim):
                    self.x_simu[0,i,j] = scaled_encoded_variables[start_iteration+i,j]
            # Make npred predictions
            for ipred in range(npred):
                start_iteration_prediction=start_iteration+(ipred+1)*n_steps
                
                y_simu = self.simulator.predict( self.x_simu )
                simulated_in_latent_space=y_simu[:,:].reshape(n_steps,latent_dim)
                latent_solution = self.simulator_scaler.inverse_transform(simulated_in_latent_space)
                latent_solution=latent_solution.reshape((n_steps,latent_dim))
                predicted_scaled_variables = self.decoder.predict(latent_solution)
                predicted_variables = self.encoder_scaler.inverse_transform( predicted_scaled_variables )

                # Store prediction
                self.simu_variables[start_iteration_prediction:start_iteration_prediction+n_steps,:]=predicted_variables.copy()
                scaled_predicted_variables = self.encoder_scaler.transform( predicted_variables )
                encoded_predicted_variables = self.encoder.predict( scaled_predicted_variables )
                scaled_encoded_predicted_variables = self.simulator_scaler.transform( encoded_predicted_variables )
                
                self.x_simu[0,:,:]= scaled_encoded_predicted_variables.copy()


    # Compare fiducial solution with encoded_decoded solution
    # starting at iter iteration
    def compute_encoded_decoded(self):
        print( "Compute autoencoded solution")

        # Prepare real_variables for simulation
        decoded_variables = self.decoder.predict( self.encoded_variables )
        self.encode_decoded_variables = self.encoder_scaler.inverse_transform( decoded_variables )
        
    # Compare fiducial solution with encoded_decoded and simulated solution
    def plot(self,ext):
        print( "Compare fiducial with AE and simulated solutions")
        
        # Plot both
        N=len(self.labels)
        figure_index=0
        for k in range(0,N,100):
            # i=perm[k]
            i=k
            lab=self.labels[i]
            plt.figure(figure_index)
            plt.plot(self.real_variables[:,i],label='ASTEC ')
            plt.plot(self.encode_decoded_variables[:,i],label='AE ')
            plt.plot(self.simu_variables[:,i],label='SIMU ')
            plt.title(lab)
            plt.xlabel("Iteration")
            plt.ylabel(lab)
            plt.legend()
            figname='compare_'+str(figure_index)
            plt.savefig(figname+ext)

            figure_index=figure_index+1
        if os.getenv( 'asbatch', None ) != '-batch' : plt.show()      

    # Load dataset
    def load(self,csvfile,start,end,iter):
        df=pandas.read_csv(csvfile,sep=';')
        df=df.iloc[start:end,:]
        df=df[self.labels]
        nb_samples=df.shape[0]
        nb_iterations = nb_samples-iter
        self.real_variables = df.iloc[iter:iter+nb_iterations].to_numpy()

        # Prepare real_variables for simulation
        scaled_variables = self.encoder_scaler.transform( self.real_variables )
        self.encoded_variables = self.encoder.predict( scaled_variables )

######################################################
# Start
#######
test=Test()
print("Reading simulation")
test.load('all_runs_circuit.csv.zip',0,10000,1000)

print("Compute encoded_decoded")
test.compute_encoded_decoded()

print("Compute simulated")
test.compute_simulated(100)

print("Display results")
test.plot('.eps')
if os.getenv( 'asbatch', None ) != '-batch' : plt.show()
print ( "NORMAL END" )

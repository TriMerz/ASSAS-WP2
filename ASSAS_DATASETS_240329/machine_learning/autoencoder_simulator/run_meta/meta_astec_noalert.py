#!/usr/bin/env python3
import os
import sys

second_chance = "/soft/anaconda3/5.1.0/bin"
try:
    import numpy as np
except ImportError:
    if ( not "PATH" in os.environ ):
        os.environ["PATH"] = ""
    if( second_chance not in os.environ["PATH"] ) :
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "proc"))
        os.environ["PATH"] = second_chance + ":" + os.environ["PATH"]
        import astec
        try:
            astec.execute(command=sys.argv, path=os.getcwd())
        except astec.Error:
            print("Unexpected error in meta_astec_noalert.py")
        sys.exit()
    else:
        raise ImportError("Numpy module is not available")

try:
    from tensorflow import keras
except ImportError:
    print('Waiting for a Python installation containing tensorflow use a local version')
    os.system('/soft/conda3/envs/py39_IA_GPU/bin/python3 '+__file__)
    exit(0)
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import linalg as LA
import joblib

sys.path.append(os.path.join('..','build_models'))
from astools import *

import pyastec as pyas
from WithModulesAstecMain import WithModulesAstecMain

# Could be read
n_steps=10
latent_dim=10

def self_die(msg):
    print(msg)
    sys.exit(1)
#
def display_family_names(base):
    pyas.odprint(base)
    for i in range(pyas.odbase_family_number(base)):
        print(pyas.odbase_name(base,i))
_SEP=':'

#
class Hal(WithModulesAstecMain):

    def __init__(self, mdat=None):
        super().__init__(mdat)
        self.iter = 0
        self.NB_META_STEPS = 10
        self.first = True
        self.logfile = open('Hal.lst','w')

    def log(self, msg):
      self.logfile.write(msg+"\n")
      self.logfile.flush()

    def _astec_module(self, success, module_name, tbeg, tend):
        super()._astec_module(success, module_name, tbeg, tend)

# Interaction with base
    def read_variables(self):
        self.base = pyas.root_database()
        i=0
        for l in self.labels:
            self.current_variables[0,i] = get_var(self.base,l)
            i=i+1

    def put_variables(self):
        self.base = pyas.root_database()
        i=0
        t=pyas.doublep_value(self.tend)
        for l in self.labels:
            v = np.float64(self.current_variables[0,i])
            # Avoid modifying boundary conditions
            # Actually, for such BC, we should verify that they have not been modified during meta time step
            if ( not l.endswith('SECT' ) ) and ( not l.endswith('P[1]' ) ) and ( not l.endswith('STAT' ) ):
                # Basic correction
                c=v
                if 'T_gas' in l or 'T_liq' in l or 'T_wall' in l or l.endswith(':T'):
                    if v < 273.15 or v > 3000.:
                        c = 300.                        
                if 'P_steam' in l:
                    if v < 0. :
                        c = 0.
                if 'x_alfa' in l :
                    if v < 0. :
                        c = 1.0e-6
                    elif v > 1.0:
                        c = 1.- 1.0e-6
                if( v != c ):
                    self.log('Correction from '+str(v)+' to '+str(c)+' in put variable to '+l)
                    v=c
                else:
                    self.log('Set '+str(v)+' to '+l)
                put_var(self.base,l,t,v )
            i=i+1
        # After having modified primary variables, we should compute other variables


    def compute_cesar_icare(self):
        time=pyas.doublep_value(self.tbeg)
        if( self.iter < n_steps ) :
            self.log("Running CESAR-ICARE modules at time "+str(time))
            self._astec_module(self.success, "CESAR_ICARE", self.tbeg, self.tend)
            self.read_variables()
            scaled_variables = self.encoder_scaler.transform( self.current_variables )
            encoded_variables = self.encoder.predict( scaled_variables )

#           for i in range(len(self.labels)):
            self.x_simu[0,self.iter,:] = self.simulator_scaler.transform( encoded_variables )[0,:]
            self.meta_iter = 0
        else:
            self.log("Running meta-simulator at time "+str(time) )
            self.meta_step()
        self.iter = self.iter + 1

# Models
    def main_loop(self):
        while self.icont > 0:
            pyas.calc_init(self.success, self.tbeg, self.tend)
            while not pyas.boolp_value(self.success):
                pyas.calc_hot_restart(self.tbeg, self.tend)
                pyas.boolp_assign(self.success, True)
                pyas.calc_before_modules(self.first_readbase, self.tbeg, self.tend)

                self._astec_module(self.first_readbase, "READBASE"   , self.tbeg, self.tend)
                self._astec_module(self.success, "RUPUICUV"   , self.tbeg, self.tend)
                self.compute_cesar_icare()
                self._astec_module(self.success, "READBASE"   , self.tbeg, self.tend)
                self._astec_module(self.success, "DROPLET"    , self.tbeg, self.tend)
                self._astec_module(self.success, "RCSMESH"    , self.tbeg, self.tend)
                self._astec_module(self.success, "CORIUM"     , self.tbeg, self.tend)
                self._astec_module(self.success, "MEDICIS"    , self.tbeg, self.tend)
                self._astec_module(self.success, "ASCAVI"     , self.tbeg, self.tend)
                self._astec_module(self.success, "THC"        , self.tbeg, self.tend)
                self._astec_module(self.success, "CPA"        , self.tbeg, self.tend)
                self._astec_module(self.success, "PH"         , self.tbeg, self.tend)
                self._astec_module(self.success, "SOPHAEROS"  , self.tbeg, self.tend)
                self._astec_module(self.success, "SAFARI"     , self.tbeg, self.tend)
                self._astec_module(self.success, "DOSE"       , self.tbeg, self.tend)
                self._astec_module(self.success, "COVI"       , self.tbeg, self.tend)
                self._astec_module(self.success, "ISODOP"     , self.tbeg, self.tend)

                pyas.calc_after_modules(self.success, self.tbeg, self.tend)
            self.icont = pyas.tool()


    def model_filename(self,model):
        models_path=os.path.join("..","build_models")
        return os.path.join(models_path,model)

    def _computation(self):
        self.main_loop()

    def meta_step(self):
        # Serious things start just here...
        time=pyas.doublep_value(self.tbeg)
        if self.meta_iter==0:
            self.log("Making prediction at time "+str(time) )
            # Need prediction
            y_simu = self.simulator.predict( self.x_simu )
            self.x_simu[0,:,:]=y_simu[:,:].reshape(n_steps,latent_dim)
        self.log("Using simulated solution at time "+str(time))
        simulated_in_latent_space = self.x_simu[0,self.meta_iter,:]
        simulated_in_latent_space=simulated_in_latent_space.reshape((1,latent_dim))
        latent_solution = self.simulator_scaler.inverse_transform(simulated_in_latent_space)
        latent_solution=latent_solution.reshape((1,latent_dim))
        predicted_scaled_variables = self.decoder.predict(latent_solution)
        self.current_variables = self.encoder_scaler.inverse_transform( predicted_scaled_variables )
        self.put_variables()
        self.meta_iter = self.meta_iter + 1
        if( self.meta_iter == n_steps ) :
          # Revert standard mode
          self.iter = 0
          self.meta_iter = 0

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


if __name__ == "__main__":
    computation = Hal("restart.mdat")
    # First read encoder, decoder and simulator
    computation.read_models()

    computation.run()
print ( "NORMAL END" )

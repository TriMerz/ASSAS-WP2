#!/usr/bin/env python3

"""--------------------------------------------------------------------------
   =======================================================================
   This script is used to read a given folder containing the binary files.
                       Only for CESAR_IO family.
           (more general code will be implemented in the future)
   =======================================================================
--------------------------------------------------------------------------"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

os.system('cls' if os.name == 'nt' else 'clear')

ASTECROOT = "/opt/astecV3.1.2/"   # astec main root path
COMPUTER  = "linux_64"  # it depends on your os, if windows use: "win64"
COMPILER  = "gccloc"    # it depends on your compiler, if windows use: "msvc"
bin_data = "/data/savini/ASSAS/WP2/ASSAS-WP2/ASSAS_DATASET_241014/PWR1300-LIKE_ASSAS/STUDY/TRANSIENT/BASE_SIMPLIFIED/LOCA/6I_CL/mycesar_io.bin"

# Start ASTEC environment
try: 
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
   print("Successfully Started the ASTEC environment!")
except ModuleNotFoundError:
   print("Error: ASTEC not found!")
   sys.exit()



''' ===== --- Data extraction  --- ===== '''

import pyodessa as pyod
print("Successfully imported pyodessa!")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def databse_extraction(binary):
   for i, base in pyod.save_iterator(binary, t_start=None):
      if i != 0.0:
         card_cesio = len(list(base.family("CESAR_IO")))
         for c in range(card_cesio):
            fam = "CESAR_IO " + str(c)
            out = fam + ":OUTPUTS"
            conv = base.get(fam + ":CONV")
            if conv != 0:
               macrobeg = base.get(fam + ":MACROBEG")
               macroend = base.get(fam + ":MACROEND")
               dtmacro = base.get(fam + ":dtmacro")
               microbeg = base.get(fam + ":STEPBEG")
               microend = base.get(fam + ":STEPEND")
               dtmicro = base.get(fam + ":dtfluid")
               iter = base.get(out + ":iter")
               varprim = base.get(out + ":VARPRIM")
   return (None)


import h5py


def create_datadset():
    filename = 'dataset.h5'

    with h5py.File(filename, 'a') as f:
        if 'macro' not in f:
            macro_group = f.create_group('macro')
        else:
            macro_group = f['macro']

        # Aggiunta dei dataset nella cartella 'macro'
        if 'macrobeg' not in macro_group:
            macro_group.create_dataset('macrobeg', data=0.0)
        if 'macroend' not in macro_group:
            macro_group.create_dataset('macroend', data=1.0)
        if 'dtmacro' not in macro_group:
            macro_group.create_dataset('dtmacro', data=0.01)

        # Creazione del sottogruppo 'micro' dentro 'macro'
        if 'micro' not in macro_group:
            micro_group = macro_group.create_group('micro')
        else:
            micro_group = macro_group['micro']

        # Aggiunta dei dataset nella cartella 'micro'
        if 'stepbeg' not in micro_group:
            micro_group.create_dataset('stepbeg', data=0.0)
        if 'stepend' not in micro_group:
            micro_group.create_dataset('stepend', data=100.0)
        if 'dtfluid' not in micro_group:
            micro_group.create_dataset('dtfluid', data=0.1)

        # Creazione del sottogruppo 'varprim' dentro 'micro'
        if 'varprim' not in micro_group:
            varprim_group = micro_group.create_group('varprim')
        else:
            varprim_group = micro_group['varprim']

        # Aggiunta di uno scalare 'iter' e una matrice dentro 'varprim'
        if 'iter' not in varprim_group:
            varprim_group.create_dataset('iter', data=0)

        if 'matrix' not in varprim_group:
            varprim_group.create_dataset('matrix', data=np.zeros((10, 10)))
    print("Dataset HDF5 creato e aggiornato con successo.")



df1 = create_datadset()


df = databse_extraction(bin_data)


'''
from concurrent.futures import ProcessPoolExecutor
def process_save(i, base):
   if i != 0.0:
      print(type(i))
      print(type(base))

with ProcessPoolExecutor() as executor:
   futures = [executor.submit(process_save, i, base) for i, base in pyod.save_iterator(bin_data, t_start=None)]
   for future in futures:
      future.result()
'''




'''
varprim = []
dtcesar = []
dtmacro = []
elapsed_time = []
saved_database = len(os.listdir(bin_data))-2    # (-2) because there are two file where base.family("CESAR_IO") is not saved: index.bin e 0.000.bin, otherwise ASTEC crash
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
               # == GET Input ==
               cad_input = len(list(fam_name + ":INPUTS"))
               for j in range(cad_input):
                  v5 = base.get(fam_name + ":INPUTS:INPUT" + str(j))

            else:
               pass
      except ModuleNotFoundError:
         print("CESAR_IO does not exist yet!")







         # online visualization (in [%]) of the database readed.
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

   data = {'time': elapsed_time[1:], 'dtcesar': dtcesar[1:]} # !! the DataFrame begins with the second converged solution [1:] !!
   for i in range(diff.shape[1]):
      data[f'var{i}'] = diff[:, i]
   df = pd.DataFrame(data).sort_values(by='time')  # datafram sorted by time
   df.set_index('time', inplace=True)  # dataframe index set with the variable time
   column_indices = {name: j for j, name in enumerate(df.columns)}

   # The reading process is extremly slow. It needs to use ASTEC and, because of that, no parallelizations could be used.
   # That's why a cashed file is saved onece the datased has been read.
   # ===== NEW CASHED FILE =====
   with open(cache_file, 'wb') as f:
      pickle.dump(df, f)
   print("Cache file saved as:", cache_file)
'''
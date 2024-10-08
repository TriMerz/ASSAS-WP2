#!/usr/bin/env python3

"""--------------------------------------------------------------------------
   =======================================================================
   This script is used to read a given folder containing the binary files.
   =======================================================================
--------------------------------------------------------------------------"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.system('cls' if os.name == 'nt' else 'clear')

ASTECROOT = "/home/marcello/IRSN/astecV3.1.1_linux64/astecV3.1.1/"   # PUT THE MAIN ASTEC ROOT HERE
COMPUTER  = "linux_64"  # it depends on your os, if windows use: "win64"
COMPILER  = "gccloc"
cache_file = 'data_cache.pkl'
bin_data = "mycesar_io.bin"   # directory name with the binary saves

# ========== LOAD CACHED DATA or READ FROM *.bin SAVES ==========
# If the binary saves have been read yet, the ASTEC environment don't start at all!!
if os.path.exists(cache_file):
   with open(cache_file, 'rb') as f:
      df = pickle.load(f)
   print("\n", "Cache loaded successfully!")
else:
   # ===== START ASTEC ENVIRONMENT =====
   sys.path.append(os.path.join(ASTECROOT, "code","proc"))
   sys.path.append(os.path.join(ASTECROOT, "code","bin", COMPUTER + "-" + "release", "wrap_python"))
   import AstecParser
   import astec
   AP = AstecParser.AstecParser()
   AP.parsed_arguments.compiler=COMPILER
   A = astec.Astec(AP)
   A.set_environment()
   import pyastec as pa
   pa.astec_init()   # !!  ASTEC Environment STARTED  !!

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


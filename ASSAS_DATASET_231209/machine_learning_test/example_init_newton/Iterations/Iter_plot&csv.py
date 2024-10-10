#!/usr/bin/env python3

ASTECROOT = "/home/marcello/IRSN/astecV3.1.1_linux64/astecV3.1.1/"
COMPUTER  = "linux_64"
COMPILER  = "gccloc"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(ASTECROOT, "code","proc"))
sys.path.append(os.path.join(ASTECROOT, "code","bin", COMPUTER + "-" + "release", "wrap_python"))
import AstecParser
import astec
AP = AstecParser.AstecParser()
AP.parsed_arguments.compiler=COMPILER
A = astec.Astec(AP)
A.set_environment()

"""===================================
   Now PyOd environment is running.
==================================="""
import pyastec as pa
pa.astec_init()

v1 = []
v2 = []
v3 = []
v4 = []
dtmacro = []
iters = []
dtcesar = []
elapsed_time = []

bin_data = "mycesar_io.bin"

print("\n")
print("DATA EXTRACTION FROM: " + bin_data)
for s,base in pa.tools.save_iterator(bin_data, t_start=None):
   if s != 0.0:
      try:
         cardinality = len(list(base.family("CESAR_IO")))
         # == GET EVERY MACRO TIME-STEP ==
         v1 = base.get("CESAR_IO:dtmacro")
         dtmacro.append(v1)
         for i in range(cardinality):
            fam_name = "CESAR_IO " + str(i)
            # == GET ITER ==
            v2 = base.get(fam_name + ":OUTPUTS:ITER")
            iters.append(v2)
            # == GET dt ==
            v3 = base.get(fam_name + ":dtfluid")
            dtcesar.append(v3)
            # == elapsed TIME ==
            v4 = base.get(fam_name + ":STEPEND")
            elapsed_time.append(v4)
      except ModuleNotFoundError:
         print("CESAR_IO does not exist yet!")

dtmacro = np.array(dtmacro)
iters = np.array(iters)
dtcesar = np.array(dtcesar)
elapsed_time = np.array(elapsed_time)

pa.end()

data = {'time': elapsed_time[0:], 'dtcesar': dtcesar[0:], 'iters': iters[0:]}
df = pd.DataFrame(data).sort_values(by='time')
df.to_csv('iters_graph_originalsss.csv', index=False)

dforiginal = pd.read_csv('iters_graph_originalsss.csv')
# dfML = pd.read_csv('iters_graph.csv')
# dfcheat = pd.read_csv('iters_graph_2.csv')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(dforiginal['time'], dforiginal['iters'], color='black', linestyle='-', label='None')
# plt.plot(dfML['time'], dfML['iters'], color='blue', linestyle='-', label='ML-test')
# plt.plot(dfcheat['time'], dfcheat['iters'], color='red', linestyle='-', label='Conv')
plt.title('Scatter Plot of Iterations vs Time')
plt.xlabel('Time')
plt.ylabel('Iterations')
plt.grid(True)
plt.legend()
plt.show()

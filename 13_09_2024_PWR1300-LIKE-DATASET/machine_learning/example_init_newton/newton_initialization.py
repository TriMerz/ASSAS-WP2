#!/usr/bin/env python3

import pyastec as pa

# Some initializations...
print("Start python script to initialize Newton...")
if "nb_pass" not in globals():
  print("Fisrt pass in this script")
  nb_pass = 0
  time0 = None
  time1 = None
  vari0 = None
  vari1 = None
else:
  pass
nb_pass = nb_pass + 1
print("Pass number",nb_pass,"in this script")

# Get the default initial vector of primary variables
root = pa.root_database()
ROOT = pa.PyOd_base(root)
newtoini = ROOT.get("CALC_OPT:CESAR:NEWTOINI")
varprim = newtoini.get("VARPRIM")
print("Length of VARPRIM =", len(varprim) )


# Get some information in the CESAR_IO, which contains cesar inputs
# This base exist only if saving input has been activated in the datadex, with the analyzer variable 'cesar_in'
tend = ROOT.get("CESAR_IO:STEPEND")
vari = varprim[0]

# Do something. Working with number of pass in not smart because it does not take into account no convergence and restarts
# but it is for an example
if( nb_pass >= 3 ):
  print("varprim[0] before modification = ", varprim[0],flush=True)
  # New value with a linear time extrapolation
  newvari = vari1 + (vari1-vari0)/(time1-time0)*(tend-time1)
  varprim[0] = newvari
  print("varprim[0] after modification = ", varprim[0],flush=True)
else:
  # Put stat to 0 to say to cesar there is nothing to do and to save a copy
  # This is not very necessary
  newtoini.put("STAT",0)

# Shift values for next pass
time0 = time1
time1 = tend
vari0 = vari1
vari1 = vari

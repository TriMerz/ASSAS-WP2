#!/usr/bin/env python3

# ATTENTION all the packages must be imported also at the beginning of the file astec_main.py
import os
import pickle
import numpy as np
import pandas as pd
import pyastec as pa

cache_file = 'data_cache.pkl'

if "nb_pass" not in globals():
  print("Fisrt pass in this script...")
  nb_pass = 0
else:
  pass

with open(cache_file, 'rb') as f:
  df = pickle.load(f)
print("\n", "Cache loaded successfully!")

print("""\
*-----------------------------------------------------------*
""",flush=True)
print("Pass number",nb_pass, "\n")

root = pa.root_database()
ROOT = pa.PyOd_base(root)
newtoini = ROOT.get("CALC_OPT:CESAR:NEWTOINI")
varprim = newtoini.get("VARPRIM")
time_end = ROOT.get("CESAR_IO:STEPEND")

print(time_end)

if time_end in df.index:
  var = df.loc[time_end].tolist() # pa.PyOd_r1() use list object as input!! .lolist()
  var = pa.PyOd_r1(var)  # conversion to pa.PyOd_r1() object

  for i in range(len(varprim)):
    e = var[i]
    varprim[i] = e

  # newtoini.put("STAT", 1, 0)
  # newtoini.put("NEWTOINI",varprim,0)

else:
  newtoini.put("STAT", 0, 0)

# == SHIFT nb_pass ==
nb_pass += 1

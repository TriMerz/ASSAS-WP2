#!/usr/bin/env python3
import os
import sys
from os.path import join

import pandas
import numpy
import astools
from common import die
import pyastec
import joblib
import threading

# This script aims to count all the data stored in the simplified databases


# Count fields in filtered databases
irun = 0
while(True):
    test=join("..","build_basis","run_"+str(irun),"reference_filtered.bin")
    if( not os.path.isdir( test ) ):
        break
    print("Processing "+test)
    print("Number of fields in " + test + " : " + str( astools.count_fields( test, 2000. ) ) )

    test=join("..","build_basis","run_"+str(irun),"reference_reduced.bin")
    if( not os.path.isdir( test ) ):
        break
    print("Processing "+test)
    print("Number of fields in " + test + " : " + str( astools.count_fields( test, 5000. ) ) )
    # irun = irun+1
    break

print ( "NORMAL END" )

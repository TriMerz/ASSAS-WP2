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

# This script aims to collect all the data stored in the simplified databases
#  in one unique dataframe
# All the file are compressed
# All the separate extractions are done in parallel through the threading mecanism

debug=False

shift=1000
def extract_and_save(test,csv_dir,ext):
    if not os.path.isdir(csv_dir):
            os.mkdir(csv_dir)
            
    for k in range(1,1+5*shift,shift):
            
        output_file=join(csv_dir,'time_'+str(k)+'_'+str(int(k+shift))+ext)
        if not os.path.isfile(output_file):
                # Create dataframe
                mapval = astools.extractDataFrame(join(test,"reference_reduced.bin"),k*1.,k*1.+shift,1.)
                df = pandas.DataFrame(mapval)
                if( len( df.index ) > 0 ) :
                    df.to_csv(output_file,sep=';',index_label='time')
                else:
                    print(test+" output file "+output_file+" is empty !")
                    break

extension=".csv.zip"
# Extract filtered databases in dataframes using threads
t=True
irun=0
maxrun=16
nb_threads_max = 4
list_of_threads = []
while(True):
    csv_dir="extracted_reduced"+str(irun)
    test=join("..","..","build_basis","run_"+str(irun))
    if( not os.path.isdir( test ) or irun > maxrun ):
        break
    print("Processing "+test)
    if( not os.path.isdir( csv_dir ) ):
        if t:
                thread = threading.Thread(None, extract_and_save, args=[test,csv_dir,extension])
                thread.start()
                list_of_threads.append(thread)
                if ( irun + 1 ) % nb_threads_max == 0:
                    for t in list_of_threads: t.join()
                    list_of_threads=[]
        else:
                extract_and_save(test,csv_dir,extension)
    irun=irun+1

# Wait for all threads to terminate
for t in list_of_threads:
    t.join()

print ( "NORMAL END" )

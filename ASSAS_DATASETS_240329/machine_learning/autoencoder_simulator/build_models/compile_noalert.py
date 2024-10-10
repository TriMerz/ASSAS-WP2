#!/usr/bin/env python3
import os.path
import sys

import pandas
import numpy
import astec
import astools
import joblib
from common import die
import pyastec as pyas

######################################################
# Start
#######


def compile_all_runs( name, needlist, droplist ):
        all_runs=name+".csv.zip"
        if os.path.isfile(all_runs): return
        irun=0
        dataframes = list()
        first_sample=[0]
        while(True):
            filtered_dir='extracted_reduced'+str(irun)

            if not os.path.isdir(filtered_dir):
                break
            nb=0
            for csv in os.listdir( filtered_dir ):
                print("Opening "+csv+" in "+filtered_dir)
                df = pandas.read_csv( os.path.join( filtered_dir, csv ), sep=';' )
                print("  File contains "+str(len(df.index))+" rows and "+str(len(df.columns))+" columns")
                labels = df.columns

                nb=nb+len(df.index)

                read_columns=list()
                for label in labels:
                    drop=True
                    for key in needlist:
                        if key in label:
                            drop=False
                            break
                    if not drop:
                        for key in droplist:
                            if key in label:
                               drop=True
                               break

                    if not drop:
                        read_columns.append(label)
                       # print("Keep column "+label)

                    #else:         
                       # print("Drop column "+label)
                print("Keep "+str(len(read_columns))+" columns")
                df = df[read_columns]
                dataframes.append( df )

            irun = irun + 1
            print("Total number in "+filtered_dir+" is "+str(nb))
            first_sample.append( first_sample[irun-1] + nb )

        print("Concatenate all files ")
        df = pandas.concat( dataframes )

        print("Fill with 0 ")
        df.fillna( 0., inplace=True )

        print("Removing constant columns")
        print("  Initial column number is "+str(len(df.columns)))
        df = df[astools.non_constant_value_features(df)]
        print("  Final column number is "+str(len(df.columns)))
        
        print("Writing to "+all_runs)
        df.to_csv(all_runs,sep=';',index_label='time')

        labels = df.columns
        joblib.dump(first_sample, 'first_sample.sav')
        
compile_all_runs( "all_runs_circuit",
                  [ "PRIMARY","SECONDAR",f"CONNECTI(CORE_UPP):SOURCE({1-pyas.odessa_shift()}):FLOW",f"CONNECTI(DC_COLD):SOURCE({1-pyas.odessa_shift()}):FLOW","ACCUMULA","MOMENTUM" ],
                  [] )
compile_all_runs( "all_runs_containment",
                  [ "CONTAINM" ],
                  [ "FP_HEAT" ] )
## compile_all_runs( "all_runs_vessel",
##                   ["VESSEL"],
##                   ["FI","FD","FE","FU"] )

print ( "NORMAL END" )

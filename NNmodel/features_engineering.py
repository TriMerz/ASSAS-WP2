#!/usr/bin/env python3

"""-----------------------------------------------------------------------
    =================================================================
    This script is used to modify the database of CESAR_IO variables
    =================================================================
-----------------------------------------------------------------------"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp

from readHDF import HDF5Reader, PlotVars, \
                    parallel_plot_vars

"""--------------------------------------------------------------------"""
class DataProcessor:
    def __init__(self, database):
        self.database = database

    def remove_duplicates(self):
        """
        Clean the DataFrame by removing duplicates.
        """
        df = self.database.drop_duplicates()       # Remove duplicates
        print(f"Removed {len(self.database) - len(df)} duplicates")
        return(df)
    
        




"""--------------------------------------------------------------------"""
def main(hdf5name, MACRO):

    if not os.path.exists(hdf5name):
        print(f"Error: The file {hdf5name} does not exist!")
        sys.exit()
    
    df = HDF5Reader(hdf5name).MACROtime()
    df = DataProcessor(df).remove_duplicates()
    




    # if MACRO:
    #     df = reader.MACROtime()
    #     df.to_csv(f'testMACRO_{hdf5name}.csv')
    # else:
    #     df = reader.CESARtime()
    #     df.to_csv(f'testCESAR_{hdf5name}.csv')
    
    

#//////////////////////////////////////////////////////////////////#
#/////////////////////////      Main      /////////////////////////#
#//////////////////////////////////////////////////////////////////#
if __name__ == "__main__":

    hdf5name = "dataset.h5"
    #outputdir = "original_plots"
    MACRO = True

    main(hdf5name, MACRO)
#!/usr/bin/env python3

"""-----------------------------------------------------------------------
    =================================================================
    This script is used to read a HDF5 database of CESAR_IO variables
          (more general code will be implemented in the future)
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

"""--------------------------------------------------------------------"""
def HDFstructure(database):
    with h5py.File(database, 'r') as f:
        f.visit(print)

"""--------------------------------------------------------------------"""
class HDF5Reader:
    def __init__(self, database):
        self.database = database

    def CESARtime(self):
        """
        Converts an H5 file into a pandas DataFrame.
        The varprim variables are expanded into separate columns.
        The row indices are set to the values of microend (dtCESAR)
        Returns:
            pd.DataFrame: DataFrame containing the micro time-step data
        """
        with h5py.File(self.database, 'r') as f:
            micro_group = f['MACRO']['MICRO']
            varprim_matrix = micro_group['VARPRIM']['varprim'][:]
            microend = micro_group['microend'][:]
            
            data_dict = {}
            for i in range(varprim_matrix.shape[1]):
                data_dict[f'var_{i}'] = varprim_matrix[:, i]
            df = pd.DataFrame(data_dict, index=microend)
            print(f'File {self.database} has been read and converted into a Pandas DataFrame')
            return(df)
    
    def MACROtime(self):
        """
        Converts an H5 file into a pandas DataFrame.
        The varprim variables are expanded into separate columns.
        The row indices are set to the values of macroend.
        Returns:
            pd.DataFrame: DataFrame containing the macro time-step data
        """
        with h5py.File(self.database, 'r') as f:
            OnlyMACRO_group = f['OnlyMACRO']
            varprim_matrix = OnlyMACRO_group['MACROvarprim'][:]
            macroend = OnlyMACRO_group['macroend'][:]
            
            data_dict = {}
            for i in range(varprim_matrix.shape[1]):
                data_dict[f'var_{i}'] = varprim_matrix[:, i]
            df = pd.DataFrame(data_dict, index=macroend)
            print(f'File {self.database} has been read and converted into a Pandas DataFrame')
            return(df)
        
"""--------------------------------------------------------------------"""
class PlotVars:
    def __init__(self, df, outputdir):
        self.df = df
        self.outputdir = outputdir
        self.plot_vars()

    def plot_vars(self):
        """
        Plots 10 random variables from the DataFrame as separate subplots (non-parallelized).
        The random selection is seeded to ensure the same variables are plotted every time.
        Returns:
            None
        """

        if not os.path.exists(self.outputdir):
            os.mkdir(self.outputdir)
        
        np.random.seed(42)  # Seed for reproducibility
        random_cols = np.random.choice(self.df.columns, size=10, replace=False) 
        for col in random_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.df.index, self.df[col], linewidth=0.1, c='blue')
            ax.scatter(self.df.index, self.df[col], s=3, c='blue', marker='x', linewidths=0.1)
            ax.set_title(f'Plot of {col}')
            ax.set_xlabel('Index')
            ax.set_ylabel(col)
            output_path = os.path.join(self.outputdir, f'{col}.svg')
            fig.savefig(output_path, format='svg')
            plt.close(fig)
    
    def plot_single_var(self, col):
        """
        Plots a single variable from the DataFrame for parallel execution.
        Args:
            col (str): The name of the variable to plot.
        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df.index, self.df[col], linewidth=0.1, c='blue')
        ax.scatter(self.df.index, self.df[col], s=3, c='blue', marker='x', linewidths=0.1)
        ax.set_title(f'Plot of {col}')
        ax.set_xlabel('Index')
        ax.set_ylabel(col)
        output_path = os.path.join(self.outputdir, f'{col}.svg')
        fig.savefig(output_path, format='svg')
        print(f'Plot of {col} has been saved to {output_path}')
        plt.close(fig)

"""--------------------------------------------------------------------"""
def parallel_plot_vars(df, outputdir, max_processes=None):
    """
    Parallelizes the plotting of each variable in the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        outputdir (str): Directory to save the plots.
        max_processes (int, optional): Maximum number of processes to use.
    """
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    plotter = PlotVars(df, outputdir)
    processes = max_processes or min(os.cpu_count() // 2, 96)
    pool = mp.Pool(processes=processes)
    tasks = [pool.apply_async(plotter.plot_single_var, args=(col,)) for col in df.columns]  # Create tasks for each column in the DataFrame
    [task.get() for task in tasks]
    pool.close()
    pool.join()

"""--------------------------------------------------------------------"""
#def parallel_h5_to_dataframe(h5filename):
#    return h5_to_dataframe(h5filename)

"""--------------------------------------------------------------------"""
def main(hdf5name, outputdir, MACRO, test, max_processes=None):
    reader = HDF5Reader(hdf5name)

    if not os.path.exists(hdf5name):
        print(f"Error: The file {hdf5name} does not exist!")
        sys.exit()
    
    if test:
        if MACRO:
            outputdir = f'testMACRO_{outputdir}'
            df = reader.MACROtime()
            print(df.head(20))
            PlotVars(df, outputdir)
        else:
            outputdir = f'testCESAR_{outputdir}'
            df = reader.CESARtime()
            print(df.head(20))
            PlotVars(df, outputdir)
    else:
        if MACRO:
            outputdir = f'MACRO_{outputdir}'
            df = reader.MACROtime()
            print(df.head(20))
            parallel_plot_vars(df, outputdir, max_processes)
        else:
            outputdir = f'CESAR_{outputdir}'
            df = reader.CESARtime()
            print(df.head(20))
            parallel_plot_vars(df, outputdir, max_processes)

#//////////////////////////////////////////////////////////////////#
#/////////////////////////      Main      /////////////////////////#
#//////////////////////////////////////////////////////////////////#
if __name__ == "__main__":

    hdf5name = "dataset.h5"
    outputdir = "original_plots"
    MACRO = True
    test = True

    main(hdf5name, outputdir, MACRO, test, max_processes=64)


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
import numpy as np
import pandas as pd
import h5py

def ASTECRun(ASTECROOT, COMPUTER, COMPILER):
   """
   Avvia l'ambiente ASTEC per l'estrazione dei dati.
    
   Args:
      ASTECROOT (str), COMPUTER (str), COMPILER (str):
      Percorso principale di ASTEC, sistema operativo e compilatore utilizzato
        
   Returns:
      None
   """
   try:
      sys.path.append(os.path.join(ASTECROOT, "code","proc"))
      sys.path.append(os.path.join(ASTECROOT, "code","bin", COMPUTER + "-" + "release", "wrap_python"))
      import AstecParser # type: ignore
      import astec # type: ignore
      AP = AstecParser.AstecParser()
      AP.parsed_arguments.compiler=COMPILER
      #AP.parsed_arguments.omp = 6
      A = astec.Astec(AP)
      A.set_environment()
      import pyastec as pa # type: ignore
      pa.astec_init()
      print("Successfully Started the ASTEC environment!")
   except ModuleNotFoundError:
      print("Error: ASTEC not found!")
      sys.exit()

def databse_extraction(binary, saved_database, pyod):
   """
   Estrae i dati da un database (directory) binario di ASTEC.
    
   Args:
      binary (str), saved_database (int), pyod (module):
      Percorso del database binario, numero di file salvati e modulo pyodessa
        
   Returns:
      pd.DataFrame, pd.DataFrame(macro_only): DataFrame contenente i dati estratti
   """
   n = 0
   data_list = {
   'macrobeg': [],
   'macroend': [],
   'dtmacro': [],
   'microbeg': [],
   'microend': [],
   'dtmicro': [],
   'iter': [],
   'varprim': []
   }

   macro_data = {
      'macrobeg': [],
      'macroend': [],
      'dtmacro': [],
      'MACROvarprim': []
   }

   for i, base in pyod.save_iterator(binary, t_start=None):
      if i != 0.0:
         n += 1
         card_cesio = len(list(base.family("CESAR_IO")))
         for c in range(card_cesio):
            fam = "CESAR_IO " + str(c)
            out = fam + ":OUTPUTS"
            conv = base.get(fam + ":CONV")
            if conv != 0:
               macrobeg = np.array(base.get(fam + ":MACROBEG")).item()
               macroend = np.array(base.get(fam + ":MACROEND")).item()
               dtmacro = np.array(base.get(fam + ":dtmacro")).item()
               microbeg = np.array(base.get(fam + ":STEPBEG")).item()
               microend = np.array(base.get(fam + ":STEPEND")).item()
               dtmicro = np.array(base.get(fam + ":dtfluid")).item()
               itera = np.array(base.get(out + ":ITER")).item()
               varprim = np.array(base.get(out + ":VARPRIM"))
               if c == card_cesio - 1:
                  MACROvarprim = varprim
                  macro_data['macrobeg'].append(macrobeg)
                  macro_data['macroend'].append(macroend)
                  macro_data['dtmacro'].append(dtmacro)
                  macro_data['MACROvarprim'].append(MACROvarprim)

               data_list['macrobeg'].append(macrobeg)
               data_list['macroend'].append(macroend)
               data_list['dtmacro'].append(dtmacro)
               data_list['microbeg'].append(microbeg)
               data_list['microend'].append(microend)
               data_list['dtmicro'].append(dtmicro)
               data_list['iter'].append(itera)
               data_list['varprim'].append(varprim)
               
         process_progression(n, saved_database)
   df = pd.DataFrame(data_list, index=data_list['microend'])
   macrodf = pd.DataFrame(macro_data, index=macro_data['macroend'])
   print(f'File {binary} has been readed and saved in Pandas DataFrame')
   return(df, macrodf)


def create_dataset(h5filename, df, macrodf):
   """
   Crea un file HDF5 contenente i dati estratti.
    
   Args:
      h5filename (str), df (pd.DataFrame), macrodf (pd.DataFrame):
      Percorso del file H5 da creare, DataFrame contenente i dati estratti e DataFrame contenente i dati macro
        
   Returns:
      None - il database viene salvato nella directory corrente
   """
   with h5py.File(h5filename, 'a') as f:
      # adding 'OnlyMACRO' as a group
      if 'OnlyMACRO' not in f:
         onlymacro_group = f.create_group('OnlyMACRO')
      else:
         onlymacro_group = f['OnlyMACRO']
      for column in ['macrobeg', 'macroend', 'dtmacro', 'MACROvarprim']:
         if column not in onlymacro_group:
            onlymacro_group.create_dataset(column, data=np.stack(macrodf[column].values))

      # adding 'macro' as a group
      if 'MACRO' not in f:
         macro_group = f.create_group('MACRO')
      else:
         macro_group = f['MACRO']
      for column in ['macrobeg', 'macroend', 'dtmacro']:
         if column not in macro_group:
            macro_group.create_dataset(column, data=np.array(df[column].values, dtype=np.float64))

      # adding 'micro' inside 'macro' as a group
      if 'MICRO' not in macro_group:
         micro_group = macro_group.create_group('MICRO')
      else:
         micro_group = macro_group['MICRO']
      for column in ['microbeg', 'microend', 'dtmicro']:
         if column not in micro_group:
            micro_group.create_dataset(column, data=np.array(df[column].values, dtype=np.float64))

      # adding 'varprim' inside 'micro' as a group
      if 'VARPRIM' not in micro_group:
         varprim_group = micro_group.create_group('VARPRIM')
      else:
         varprim_group = micro_group['VARPRIM']
      if 'iter' not in varprim_group:
         varprim_group.create_dataset('iter', data=np.array(df['iter'].values, dtype=np.float64))
      if 'varprim' not in varprim_group:                                # Handle 'varprim' separately
         varprim_group.create_dataset('varprim', data=np.stack(df['varprim'].values))


def process_progression(nb, saved_database):
   if (nb%10 == 0):
      progress = nb/saved_database * 100
      print(f"Reading... {progress:.2f}% complete")
   return(None)


"""-----------------------
=== = - -  Main  - - = ===
-----------------------"""
def main():
   ASTECROOT = "/opt/astecV3.1.2/"   # astec main root path
   COMPUTER  = "linux_64"  # it depends on your os, if windows use: "win64"
   COMPILER  = "gccloc"    # it depends on your compiler, if windows use: "msvc"

   ASTECRun(ASTECROOT, COMPUTER, COMPILER)
   try:
      import pyodessa as pyod # type: ignore
      print("Successfully imported pyodessa!")
   except ImportError:
      print("Error: pyodessa not found!")
      sys.exit()

   hdf5name = "dataset.h5"
   bin_data = "/data/savini/ASSAS/WP2/ASSAS-WP2/ASSAS_DATASET_241014/PWR1300-LIKE_ASSAS/STUDY/TRANSIENT/BASE_SIMPLIFIED/LOCA/6I_CL/mycesar_io.bin"
   if not os.path.exists(bin_data):
      print(f"Error: The directory {bin_data} does not exist!")
      sys.exit()
   saved_database = len(os.listdir(bin_data))-2    # (-2) because there are two file where base.family("CESAR_IO") is not saved: index.bin e 0.000.bin, otherwise ASTEC crash
   
   df, macrodf = databse_extraction(bin_data, saved_database, pyod)
   create_dataset(hdf5name, df, macrodf)

if __name__ == "__main__":
    main()
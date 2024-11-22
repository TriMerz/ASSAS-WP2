#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import pandas as pd
import h5py

class AstecDataExtractor:
    def __init__(self, astec_root, computer, compiler):
        """
        Initialize ASTEC environment.
        
        Args:
            astec_root (str): Main ASTEC root path
            computer (str): Operating system (e.g., 'linux_64', 'win64')
            compiler (str): Compiler used (e.g., 'gccloc', 'msvc')
        """
        self.astec_root = astec_root
        self.computer = computer
        self.compiler = compiler
        self.pyastec = None
        self.start_astec_environment()

    def start_astec_environment(self):
        """Start the ASTEC environment and initialize necessary modules."""
        try:
            sys.path.append(os.path.join(self.astec_root, "code", "proc"))
            sys.path.append(os.path.join(self.astec_root, "code", "bin", 
                                       self.computer + "-" + "release", "wrap_python"))
            import AstecParser # type: ignore
            import astec # type: ignore
            
            AP = AstecParser.AstecParser()
            AP.parsed_arguments.compiler = self.compiler
            AP.parsed_arguments.omp = 6
            A = astec.Astec(AP)
            A.set_environment()
            import pyastec as pa # type: ignore
            pa.astec_init()
            self.pyastec = pa
            print("Successfully Started the ASTEC environment!")
        except ModuleNotFoundError:
            print("Error: ASTEC not found!")
            sys.exit()

    def databse_extraction(self, binary, saved_database, pyod):
        """
        Extract data from an ASTEC binary database directory.
        
        Args:
            binary (str): Path to binary database
            saved_database (int): Number of saved files
            pyod (module): pyodessa module
            
        Returns:
            tuple: (pd.DataFrame, pd.DataFrame) containing extracted data and macro-only data
        """
        n = 0
        data_list = {
            'macrobeg': [], 'macroend': [], 'dtmacro': [],
            'microbeg': [], 'microend': [], 'dtmicro': [],
            'iter': [], 'varprim': []
        }
        
        macro_data = {
            'macrobeg': [], 'macroend': [], 'dtmacro': [], 'MACROvarprim': []
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
                
                self._process_progression(n, saved_database)
                
        df = pd.DataFrame(data_list, index=data_list['microend'])
        macrodf = pd.DataFrame(macro_data, index=macro_data['macroend'])
        print(f'File {binary} has been read and saved in Pandas DataFrame')
        return df, macrodf

    def create_dataset(self, h5filename, df, macrodf):
        """
        Create an HDF5 file containing the extracted data.
        
        Args:
            h5filename (str): Path to create H5 file
            df (pd.DataFrame): DataFrame containing extracted data
            macrodf (pd.DataFrame): DataFrame containing macro-only data
        """
        with h5py.File(h5filename, 'a') as f:
            # OnlyMACRO group
            if 'OnlyMACRO' not in f:
                onlymacro_group = f.create_group('OnlyMACRO')
            else:
                onlymacro_group = f['OnlyMACRO']
            for column in ['macrobeg', 'macroend', 'dtmacro', 'MACROvarprim']:
                if column not in onlymacro_group:
                    onlymacro_group.create_dataset(column, data=np.stack(macrodf[column].values))

            # MACRO group
            if 'MACRO' not in f:
                macro_group = f.create_group('MACRO')
            else:
                macro_group = f['MACRO']
            for column in ['macrobeg', 'macroend', 'dtmacro']:
                if column not in macro_group:
                    macro_group.create_dataset(column, data=np.array(df[column].values, dtype=np.float64))

            # MICRO group
            if 'MICRO' not in macro_group:
                micro_group = macro_group.create_group('MICRO')
            else:
                micro_group = macro_group['MICRO']
            for column in ['microbeg', 'microend', 'dtmicro']:
                if column not in micro_group:
                    micro_group.create_dataset(column, data=np.array(df[column].values, dtype=np.float64))

            # VARPRIM group
            if 'VARPRIM' not in micro_group:
                varprim_group = micro_group.create_group('VARPRIM')
            else:
                varprim_group = micro_group['VARPRIM']
            if 'iter' not in varprim_group:
                varprim_group.create_dataset('iter', data=np.array(df['iter'].values, dtype=np.float64))
            if 'varprim' not in varprim_group:
                varprim_group.create_dataset('varprim', data=np.stack(df['varprim'].values))

    def _process_progression(self, nb, saved_database):
        """Show progress of data extraction."""
        if (nb % 10 == 0):
            progress = nb/saved_database * 100
            print(f"Reading... {progress:.2f}% complete")

    def close(self):
        """Properly close the ASTEC environment."""
        if self.pyastec:
            self.pyastec.end()
            print("ASTEC environment closed successfully.")



def main():
    ASTECROOT = "/opt/astecV3.1.2/"
    COMPUTER = "linux_64"
    COMPILER = "gccloc"
    
    # Initialize extractor
    extractor = AstecDataExtractor(ASTECROOT, COMPUTER, COMPILER)
    
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
    
    saved_database = len(os.listdir(bin_data)) - 2
    
    try:
        df, macrodf = extractor.databse_extraction(bin_data, saved_database, pyod)
        extractor.close()
        extractor.create_dataset(hdf5name, df, macrodf)
    except:
        print("Error: An exception occurred!")
        sys.exit()
        

if __name__ == "__main__":
    main()
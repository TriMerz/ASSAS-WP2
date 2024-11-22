#!/usr/bin/env python3
'''This script cleans batteries directories after the full run of "machine_learning" testlist'''
import os
import glob
import shutil

def last_line(file, n=-100):
    ''' Getting last line of input file '''
    f = open(file,'rb')
    try:
        f.seek(n,2)
    except IOError:
        f.seek(0)
    line = f.read()
    line = str(line, encoding='iso-8859-1')
    f.close()
    return line


if __name__ == "__main__":
    # List of files to parse for checking result
    calclist = [os.path.join("SG_metamodel", "astec-simulators", "astec_meta_steam_generators_noalert.res"),
                os.path.join("filter_consistency", "filter_consistency_noalert.res"),
                os.path.join("autoencoder_simulator", "build_models", "compare_prediction_with_ASTEC_noalert.res")]

    # Checking last line of .res files
    everything_ok = True
    for calc in calclist:
        lastline = last_line(calc)
        done = ('NORMAL END' in lastline) and ('ABNORMAL END' not in lastline)
        everything_ok = (everything_ok and done)

    if everything_ok:
        # List of directories to remove
        patterns = [os.path.join("build_basis", "run_*") + os.sep,
                    os.path.join("autoencoder_simulator", "build_models", "extracted_reduced*") + os.sep]

        # Deleting directories only matching pattern
        for pattern in patterns:
            paths = glob.glob(pattern)
            for path in paths:
                shutil.rmtree(path)
                print(f"Deleting {path}")

        print("NORMAL END")
    else:
        print("At least one calculation has not finished well")
        print("ABNORMAL END")

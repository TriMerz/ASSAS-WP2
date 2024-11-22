#!/usr/bin/env python3
import os
import sys
from os.path import join, isfile, isdir
from extract_steam_generators_data import extract

# This script aims to collect the data stored in the simplified databases
#  relative to steam generator
# All the file are compressed

dir_pattern = join( "..", "..", "build_basis", "run_*", "reference_reduced.bin" )
output_folder = 'sg_model'
output_file_prefix = 'extracted'
batch_size = 1000

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

extract(dir_pattern, output_folder, output_file_prefix, batch_size)

ok=False
if( isdir( output_folder ) ):
    if( isfile( join( output_folder, 'metadata.pkl' ) ) ):
        ok = True

if(ok) :
    print ( "NORMAL END" )

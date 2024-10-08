#!/usr/bin/env python3

import os
import sys

dir_tcs   = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_tcs)

#exit if mode is standalone
if ("standalone" in os.environ):
  if( os.environ["standalone"] == "-standalone" ) :
    print("Computation not realised because of standalone mode detected.")
    print("NORMAL END")
    sys.exit()

# Configuration
import AstecParser
import astec
AP = AstecParser.AstecParser()
AP.set_from_environ()
AST = astec.Astec(AP)

# build emulator.exe
astec_builddir = os.environ["astec_builddir"]
AST.update("reload.cfg")

# run the executable (with reload.mdat which is hard coded)
new_astec_exe = os.path.join(astec_builddir,'emulator.exe')
if os.path.isfile(new_astec_exe):
    print("Start computation...")
    AST.astec_parser.parsed_arguments.exe = new_astec_exe
    AST.run("reload.mdat") # this filename has no impact, it is hard coded in main.cc
else:
    print("  ")
    print(" Problem during the generation of the local executable : " + new_astec_exe )
    print(" TEST FAILED")

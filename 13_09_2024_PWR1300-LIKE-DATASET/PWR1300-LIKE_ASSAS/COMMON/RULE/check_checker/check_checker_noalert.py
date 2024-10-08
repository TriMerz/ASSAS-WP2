#!/usr/bin/env python3

import os

#Recover environment arguments
root_path = os.path.dirname(os.environ['astec'])
astec_builddir = os.environ["astec_builddir"]
argbranch  = os.environ["argbranch"] if "argbranch" in os.environ else ""
compiler   = os.environ["compiler"]
dbg        = os.environ["dbg"] if "dbg" in os.environ else ""
astec_main = os.path.join(root_path,'code','proc','astec.py')

#Set cfgpath environment variable
os.environ["cfgpath"] = os.path.join(root_path,"cfg")

#Initialize results to False
check_checker_rules = False
check_builder_rules = False
check_release_rules = False
check_astec_rules = False

check = False

#Control astec rules
dataset="../assas_rules.rul"
res="check_ASSAS_rules.res"
cmd        = astec_main + " -rulcheck " + argbranch + " -compiler " + compiler + " " + dbg \
             + " " + dataset + " " + res
os.system(cmd)
f = open(res,"r",encoding='iso-8859-1')
for line in f.readlines():
    if " warnings            0  errors            0" in line :
        check = True
        break
f.close()

#Summarize tests
if( check ):
    print("NORMAL END")
else:
    print("Some errors have been detected in ASSAS rules")

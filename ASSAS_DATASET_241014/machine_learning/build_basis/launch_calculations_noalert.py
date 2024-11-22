#!/usr/bin/env python3

# Tool dedicated to calculation launching.
#

import os
from os import listdir
from os.path import *
import sys
import math
import time
import string
import shutil
from argparse import ArgumentParser

rep = os.environ['astec']
print( 'Running: ' + str( sys.argv ) )
print( '  with astec located in: '+rep )

from astec import Astec,TestList
import AstecParser
from common import die


# Utilities
_SEP=':'


def run_astec(ast,testfile_name):
        # run ASTEC
        testlist=TestList([])
        testlist += ast.get_tests(testfile_name,'.')
        ast.run_testlist(testlist)

def build_sample(plan):
    dim=len(plan)
    index={}
    n=1
    for key in plan:
        index[key]=0
        n*=len(plan[key])
    sample=[]
    for i in range(n):
        vec={}
        for key in plan:
            vec[key]=plan[key][index[key]]
        sample.append(vec)
        for key in plan:
            index[key]=index[key]+1
            if(index[key]<len(plan[key])):
                break
            index[key]=0
    return sample

def write_sample(sample,inital,basis_path,reference_path,name_of_testslist_file):

    dataset_list=[]
    calculation=0
    test_file=open(name_of_testslist_file,"w")
    for a_sample in sample:
        name="run_"+str(calculation)

        directory=os.path.join( basis_path, name)
        if( not os.path.isdir( directory ) ):
            os.mkdir( directory )

        mdat=os.path.join( directory, "reference.mdat")
        test_file.write(mdat+" ; "+name+"\n")

        for f in listdir(reference_path):
            if '.dat' in f or '.mdat' in f or '.ana' in f:
                shutil.copyfile(os.path.join(reference_path,f), os.path.join(directory,f))

        driving=os.path.join( directory, "driving.ana")
        with open(driving) as file:
           lines = [line.rstrip() for line in file]
        newfile=open(driving,"w")
        for line in lines:
           redef=False
           for key in a_sample:
             if line.startswith(key):
               newfile.write(key+" = "+str(a_sample[key])+"\n")
               redef=True
           if not redef:
               newfile.write(line+"\n")
        newfile.close()
        calculation=calculation+1

    test_file.close()


#----------------------------------------------------------------------------------------------
# MAIN
#----------------------------------------------------------------------------------------------
#
name_of_testslist_file="testslist.test"

# Parse remaining argument with ASTEC parser
AP = AstecParser.AstecParser()
AP.set_from_environ()
AP.parsed_arguments.batch = True
AP.parsed_arguments.queue = "seq"
AP.parsed_arguments.wait = 60
ast = Astec( AP )
ast.res_path = os.getcwd()

plan={ "tpesp" : [1500., 2000., 2500., 3000.], "tsrv2" : [3000., 3500., 4000., 4500.] }

sample=build_sample(plan)
print('Plan contains '+str(len(sample))+' samples')

basis_path = os.getcwd()
reference_path = os.path.abspath("reference")
initial="reference.mdat"
test_list=write_sample(sample,initial,basis_path,reference_path,name_of_testslist_file)



run_astec(ast,name_of_testslist_file)
print ( "NORMAL END" )

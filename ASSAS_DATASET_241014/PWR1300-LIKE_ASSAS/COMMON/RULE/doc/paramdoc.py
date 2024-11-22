#!/usr/bin/env python3
import glob
import os.path
import subprocess
import sys

root_path = os.path.dirname(os.environ['astec'])
lib_builddir = os.environ["lib_builddir"]

ruldoc=os.path.join(root_path,"odessa",lib_builddir,"ruldoc.exe")
if( os.path.isfile( os.path.join(root_path,"code","proc","VERSION.txt") ) ):
  with open(os.path.join(root_path,"code","proc","VERSION.txt"), "r", encoding="iso-8859-1") as f:
    version_astec = f.read()
else :
  version_astec = "V22.dev"

here = os.getcwd()
dirdoc = os.path.join(here,"online_doc")
if( not os.path.isdir(dirdoc) ):
  os.makedirs( dirdoc )
os.chdir(os.path.dirname(here))
#clean if last execution of ruldoc failed
for fic in glob.glob("*.html"):
  os.remove(fic)

print( " ------------------------------------------------------------------------" )
print( " Generation of the ASSAS documentation  : on-line manual "                 )
print( " Put on " + dirdoc                                                         )
print( " ------------------------------------------------------------------------" )


# ASTEC full user's doc
print( "  -- full ASSAS user documentation " )
f = open("param_input_file.txt","w")
f.write("0\n")
f.write("assas_param.rul\n")
f.write("0\n")
f.write("html\n")
f.write("assas_parameters.html\n")
f.write("ASTEC " + version_astec + " on-line ASSAS parameters manual\n")
f.close()
f = open("param_input_file.txt","r")
process = subprocess.Popen(ruldoc, shell=False, stdin=f, stdout=subprocess.PIPE, universal_newlines=True)
stdout,stderr = process.communicate()
f.close()
os.rename("info", "infoall")

#clean
os.remove("param_input_file.txt")
os.remove("infoall")
for fic in glob.glob(os.path.join(dirdoc,"*.html")):
  if( os.path.basename(fic) != "index.html" ):
    os.remove(fic)
#store .html
for fic in glob.glob("*.html"):
  os.rename(fic,os.path.join(dirdoc,fic))
#back to initial directory
os.chdir(here)

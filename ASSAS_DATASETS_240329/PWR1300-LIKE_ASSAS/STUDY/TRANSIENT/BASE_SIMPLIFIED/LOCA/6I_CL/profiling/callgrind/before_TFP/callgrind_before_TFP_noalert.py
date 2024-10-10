#!/usr/bin/env python3

import os
import subprocess
import pyastec as pa

pa.astec_init()

# Configuration
import AstecParser
AP = AstecParser.AstecParser()
AP.set_from_environ()
args      = AP.rebuild_command_line_arguments()
#Adding callgrind option
args     += " -callgrind -functions='astec_calc_and_tool' "
root_path = os.path.dirname(os.environ['astec'])
astec_py  = os.path.join(root_path,"code","proc","astec.py")

saving_dir = "../../../LOCA_6I_CL_1300_LIKE_SIMPLIFIED_ASSAS.bin"

#Creating param.dat file to set restart time
#Getting TFP
base = pa.restore(saving_dir, pa.tools.get_list_of_saving_time_from_path(saving_dir)[-1])
tfp = base.get("SEQUENCE:TFP")
restart_time = tfp-2000.

#Getting correct value of restart_time and corresponding time_step
base = pa.restore(saving_dir, restart_time)
step = base.get("SEQUENCE:STEP")
restart_time = float(int(base.get("LOADTIME")))
tstop = restart_time + 4.*step
with open("param.dat", 'w') as out:
    out.write("(restart_time = "+str(restart_time)+")\n")
    out.write("(tstop = "+str(tstop)+")\n")
    out.flush()

# name of the input file
input_file = "callgrind_before_TFP.dat"
output_file = "callgrind_before_TFP.res"

# then let's run the test case
# for that, we use subprocess
cmd = astec_py + " " + args +  " " + input_file + " " + output_file
process = subprocess.Popen(cmd, shell=True, cwd=os.getcwd())

# Ensure that calculation is finished
output = process.communicate()[0]

f = open(output_file,'rb')
try : f.seek(-1000,2)
except IOError : f.seek(0)
line = f.read()
line = str(line, encoding='iso-8859-1')
f.close()

if "NORMAL END" in line:
    print("NORMAL END")
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
       if "callgrind.out" in f:
           subprocess.call(["chmod", "640", f])
else:
    print("Something went wrong during restart calculation")
    print("-> see "+output_file+" for more information")
    print("Abnormal termination")

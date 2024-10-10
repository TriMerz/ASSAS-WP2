#!/usr/bin/env python3

#This script shows one way to generate several calculations by modifying a driving parameter in a certain interval
#The general command is
#./generate_campaign.py -config config.dat -outpudir output_name

#By using this command, the parameter assas_param is considered to be valid on [min_value, max_value] interval and in this study, nb_cal calculations are launched, assas_param interval is divided in homogeneous value steps.

from argparse import ArgumentParser
import os
import sys
import shutil

class Config:
    """
    Class representing the configuration of the study
    """
    def __init__(self,config_file, output):
        self.config_dir=os.path.abspath(os.path.dirname(config_file))
        self.listval = []
        for line in open(config_file, "r", encoding='iso-8859-1'):
            if "pwr1300_path" in line:
                self.pwr1300 = line.split("=")[1].strip()
            elif "scenario" in line:
                self.scenario = line.split("=")[1].strip()
            elif "parameter" in line:
                self.param=line.split("=")[1].strip()
            elif "min_value" in line:
                self.min_val = float(line.split("=")[1].strip())
            elif "max_value" in line:
                self.max_val = float(line.split("=")[1].strip())
            elif "nb_calc" in line:
                self.nb_calc = int(line.split("=")[1].strip())
            else:
                sys.exit("Unknown configuration parameter"+ line.split("=")[0].strip())

        if not os.path.isdir(self.pwr1300):
            print (" ERROR : " + self.pwr1300 + " does not exist !!")
            sys.exit()
        if output == None:
            self.outputdir = os.path.join(os.getcwd(), self.param)
        else:
            self.outputdir = output

        #Getting reference folder depending on scenario
        if self.scenario == 'LOCA':
            self.refdir =os.path.join(self.config_dir, "STUDY", "TRANSIENT", "BASE_SIMPLIFIED", "LOCA", "6I_CL")
        elif(self.scenario == 'SBO'):
            self.refdir =os.path.join(self.config_dir, "STUDY", "TRANSIENT", "BASE_SIMPLIFIED", "SBO", "SBO_feedbleed")
        else:
            sys.exit("Unknown scenario")


        #Calculating all values of current parameter
        step = (self.max_val - self.min_val)/(self.nb_calc-1)
        for i in range(self.nb_calc):
            self.listval.append(self.min_val + i*step)

    def create_folders_and_testlist(self):
        os.makedirs(self.outputdir)

        testlist = open(os.path.join(self.outputdir, self.param+".test"), 'w', encoding='iso-8859-1')
        for count, val in enumerate(self.listval):
            dirname = self.param+"_"+str(count)
            os.makedirs(os.path.join(self.outputdir, dirname))
            for data in os.listdir(self.refdir):
                if os.path.splitext(data)[1] == ".mdat": #copying data input and adding path to pwr1300 at the begin
                    with open(os.path.join(self.refdir, data), 'r', encoding='iso-8859-1') as original:
                        data_input = original.read()
                    with open(os.path.join(self.outputdir, dirname, data), 'w', encoding='iso-8859-1') as final:
                        final.write("(path1300 = \""+self.pwr1300+"\")\n"+data_input)
                        final.flush()
                    mdat = os.path.join(dirname, data)
                elif data == "driving.ana":
                    with open(os.path.join(self.refdir, data), 'r', encoding='iso-8859-1') as original:
                        data_input = original.read()
                    with open(os.path.join(self.outputdir, dirname, data), 'w', encoding='iso-8859-1') as final:
                        final.write(data_input + "\n" + self.param + " = " + str(val))
                        final.flush()
                elif os.path.splitext(data)[1] == ".dat":
                    shutil.copy(os.path.join(self.refdir, data), os.path.join(self.outputdir, dirname, data))
                else:
                    #Other extension are ignored
                    pass
            testlist.write(mdat+";"+self.param+str(count)+"\n")
        testlist.flush()
        testlist.close()


# ==============================================================================
# Main procedure
# ==============================================================================

if __name__ == "__main__":

    #Argument parser definition
    parser = ArgumentParser(description="This script generates a tree of caluclation depending on input parameter and number of wanted calcultion")
    parser.add_argument('-config', help="Absolute/relative path to configuration file")
    parser.add_argument('-outputdir', help="Path of the output folder",default=None)

    #Parsing arguments
    [parsed_arguments,additional_arguments]= parser.parse_known_args()
    config_path   = os.path.abspath(parsed_arguments.config)
    outputdir = parsed_arguments.outputdir
    if outputdir != None:
        outputdir = os.path.abspath(outputdir)

    #Checking consistency of given arguments
    if outputdir != None and os.path.isdir(outputdir):
        print (" ERROR : " + outputdir + " already exists !!")
        sys.exit()

    # ================
    # First launch sensitivity campain
    # ================
    config=Config(config_path, outputdir)
    config.create_folders_and_testlist()

    #config.generateset()

    #create_campaign(outputdir, param, nb_launch, min_val, max_val)
    
    #tex_list = os.path.abspath(parsed_arguments.tex_list)


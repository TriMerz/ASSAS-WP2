#!/usr/bin/env python3
import os
import astec
import logging.config
import fluent.fill as fill
import fluent.comparator as flca
from fluent.astec import LOADTIME
import fluent.astec.sequence as flseq
from fluent.astec import ROOT
import pyastec as pyas
import logging
import itertools

CALC_DIR = os.path.join( "..","build_basis","run_0" )

REFERENCE = os.path.join( CALC_DIR , "reference.bin" )
MERGED = "merged.bin"
OUTPUT_REFERENCE_FILTERED = "output.bin"
OUTPUT_FILTERED = "output_filtered.bin"
SINGLE_STEP_FILE = "single_step.mdat"

LOADTIME_DIFFERENCE_DELTA = 1.0
IGNORE = [ "BASE:CALC_OPT[0]:ICARE[0]:DTlast[0]" , "TEMP0", "PORO" ]

exit_code = 0
__cpt__=0

def run(file:str):
    global __cpt__
    AP = astec.AstecParser.AstecParser()
    AP.set_from_environ()
    AP.parsed_arguments.batch = True
    ast = astec.Astec(AP)
    ast.res_path = os.getcwd()
    ast.run(file, output_file="res"+str(__cpt__))
    __cpt__=__cpt__+1    

def check_loadtime(base:pyas.odbase, expected_loadtime:float):
    actual = LOADTIME[base]
    if abs(actual - expected_loadtime) > LOADTIME_DIFFERENCE_DELTA:
        logging.error(f"Loadtime {actual} doesn't match expected {expected_loadtime}")
        exit(1)
    else:
        logging.info(f"Loadtime match : actual {actual} close enough to expected {expected_loadtime}")

def merge(garbage:pyas.odbase, filtered:pyas.odbase):
    fill.fill(ROOT, garbage, filtered)

    time = flseq.TIME[filtered]
    step = flseq.STEP[filtered]
    tima = flseq.TIMA[filtered]
    iter = flseq.ITER[filtered]

    logging.info(f"Merging : time={time}, step={step}, sum={time+step}")

    LOADTIME[garbage] = time + step
    flseq.TIMA[garbage] = tima
    flseq.ITER0[garbage] = iter
    flseq.LOOKUNIT[garbage] = 0
    flseq.CPUTIME[garbage] = 0
    
    return garbage

def test_case(test_time:float, garbage_time:float, filter_file:str):
    global exit_code
    logging.info(f"Test case : filter={filter_file}, time={test_time}, garbage={garbage_time}")
    

    logging.info(f"Running single step on reference, actual test time = {test_time}")

    os.environ["output"] = OUTPUT_REFERENCE_FILTERED
    os.environ["filter"] = filter_file
    os.environ["file_restart"] = REFERENCE
    os.environ["time_restart"] = str(float(test_time))
    run(SINGLE_STEP_FILE)

    logging.info(f"Loading 2 odessa bases : {REFERENCE}:{garbage_time}, {OUTPUT_REFERENCE_FILTERED}:{test_time}")
    garbage = pyas.odloaddir(REFERENCE, garbage_time)
    filtered = pyas.odloaddir(OUTPUT_REFERENCE_FILTERED, test_time)

    check_loadtime(garbage, garbage_time)
    check_loadtime(filtered, test_time)

    logging.info(f"Performing merge")
    merged = merge(garbage, filtered)

    merged_loadtime = LOADTIME[merged]

    logging.info(f"Saving merged at actual loadtime : {merged_loadtime}")

    pyas.odsavedir(merged, MERGED, merged_loadtime, pyas.odbase_init())
    
    logging.info(f"Running single step from merged file at time {merged_loadtime}")
    os.environ["output"] = OUTPUT_FILTERED
    os.environ["filter"] = filter_file
    os.environ["file_restart"] = MERGED
    os.environ["time_restart"] = str(float(test_time))
    run(SINGLE_STEP_FILE)

    time_compare = test_time + 1

    logging.info(f"Comparing at time {time_compare}")

    reference_filtered = pyas.odloaddir(OUTPUT_REFERENCE_FILTERED, time_compare)
    output_filtered = pyas.odloaddir(OUTPUT_FILTERED, time_compare)

    check_loadtime(reference_filtered, time_compare)
    check_loadtime(output_filtered, time_compare)
    
    logging.info(f"Loaded 2 odessa bases : {OUTPUT_REFERENCE_FILTERED} and {OUTPUT_FILTERED} at time {time_compare}")

    valid = True
    for diff in itertools.chain(flca.Comparator(output_filtered, reference_filtered).compare(ROOT), flca.Comparator(reference_filtered, output_filtered).compare(ROOT)):
        is_ignored=False
        for i in IGNORE:
            if i in str(diff):
                print("Ignored difference: "+str(diff))
                is_ignored = True               
                continue

        if not is_ignored:
            valid = False
            print("Relevant difference: "+str(diff))

    if valid:
        logging.info("No relevant difference found : perfect")
    else:
        logging.error("Found differences between filtered databases")
        exit_code = 1

LOGGING_FILE = "logging.conf"

if __name__ == "__main__":
    if os.path.exists(LOGGING_FILE):
        logging.config.fileConfig(LOGGING_FILE)

    filter = os.path.join( CALC_DIR, "filter.dat" )
    test_case(2000, 1000, filter )
#    test_case(1000, 2000, filter )

    if( exit_code == 0 ) : print ( "NORMAL END" )
    exit(exit_code)

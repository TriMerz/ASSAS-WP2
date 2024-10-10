#!/usr/bin/env python3

import sys
import os
second_chance = "/soft/anaconda3/5.1.0/bin"
matplotlib_is_available = False
try:
    #When launching this script on the cluster, matplotlib draw the figure in a
    #buffer if a DISPLAY is defined, even if the DISPLAY is not working. This
    #will lead to a crash with following error message:
    #QStandardPaths: XDG_RUNTIME_DIR points to non-existing path
    #'/run/user/...', please create it with 0700 permissions.
    #To ensure that no curves are drawn (but only saved), following statement
    #must be used:
    from matplotlib import use as matplotlib_use
    matplotlib_use('Agg')

    #Then matplotlib librairy can be loaded
    import matplotlib.pyplot as plt
    matplotlib_is_available = True
except:
    if ( not "PATH" in os.environ ):
        os.environ["PATH"] = ""
    if( second_chance not in os.environ["PATH"] ) :
        os.environ["PATH"] = second_chance + ":" + os.environ["PATH"]
        import astec
        try:
            astec.execute(command=sys.argv, path=os.getcwd())
        except astec.Error:
            print("Unexpected error")
        sys.exit()
    else:
        print("No curves will be drawn because matplotlib library is "
            "unavailable")
        pass

import PostproProfiling as PostPro

saving_dir = "../SBO_fb_1300_LIKE_SIMPLIFIED_ASSAS_PROFILING.bin/"
savename   = "SBO_fb_SIMPL"
times      = ['TFP', 'TRUP']

post = PostPro.Profiling(saving_dir    =saving_dir, \
                         compact_saving=None,       \
                         nb_smooth     =20,         \
                         pheno_times   =times)

post.run_profiling_tool(savename)

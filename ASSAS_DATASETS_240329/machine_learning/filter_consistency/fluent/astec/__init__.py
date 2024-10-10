import fluent.path as flpa
import pyastec as pyas

ROOT = flpa.Root()
LOADTIME = flpa.FloatPath("LOADTIME", 1 - pyas.odessa_shift(), ROOT)
VISU = flpa.BasePath("VISU", 1 - pyas.odessa_shift(), ROOT)
SAVE = flpa.BasePath("SAVE", 1 - pyas.odessa_shift(), ROOT)
CALC_OPT = flpa.BasePath("CALC_OPT", 1 - pyas.odessa_shift(), ROOT)
WARNINGS = flpa.IntPath("WARNINGS", 1 - pyas.odessa_shift(), ROOT)

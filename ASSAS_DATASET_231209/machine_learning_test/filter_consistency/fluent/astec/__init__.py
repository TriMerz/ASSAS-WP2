import fluent.path as flpa

ROOT = flpa.Root()
LOADTIME = flpa.FloatPath("LOADTIME", 0, ROOT)
VISU = flpa.BasePath("VISU", 0, ROOT)
SAVE = flpa.BasePath("SAVE", 0, ROOT)
CALC_OPT = flpa.BasePath("CALC_OPT", 0, ROOT)
WARNINGS = flpa.IntPath("WARNINGS", 0, ROOT)

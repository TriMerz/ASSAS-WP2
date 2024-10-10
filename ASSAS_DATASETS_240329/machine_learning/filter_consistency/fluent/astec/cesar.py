import fluent.path as flpa
import fluent.astec as flas
import pyastec as pyas

CESAR = flpa.BasePath("CESAR", 1 - pyas.odessa_shift(), flas.CALC_OPT)
FACTL_DL = flpa.FloatPath("factL_dL", 1 - pyas.odessa_shift(), CESAR)
STOP_FLUX = flpa.IntPath("StopFlux", 1 - pyas.odessa_shift(), CESAR)
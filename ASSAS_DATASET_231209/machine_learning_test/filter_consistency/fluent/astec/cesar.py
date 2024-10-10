import fluent.path as flpa
import fluent.astec as flas

CESAR = flpa.BasePath("CESAR", 0, flas.CALC_OPT)
FACTL_DL = flpa.FloatPath("factL_dL", 0, CESAR)
STOP_FLUX = flpa.IntPath("StopFlux", 0, CESAR)
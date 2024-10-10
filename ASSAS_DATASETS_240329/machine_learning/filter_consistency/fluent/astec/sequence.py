import fluent.path as flpa
import fluent.astec as flas
import pyastec as pyas

SEQUENCE = flpa.BasePath("SEQUENCE", 1 - pyas.odessa_shift(), flas.ROOT)
TSCRAM = flpa.FloatPath("TSCRAM", 1 - pyas.odessa_shift(), SEQUENCE)
STEP = flpa.FloatPath("STEP", 1 - pyas.odessa_shift(), SEQUENCE)

TIME = flpa.FloatPath("TIME", 1 - pyas.odessa_shift(), SEQUENCE)
TIMA = flpa.FloatPath("TIMA", 1 - pyas.odessa_shift(), SEQUENCE)
ITER0 = flpa.IntPath("ITER0", 1 - pyas.odessa_shift(), SEQUENCE)
ITER = flpa.IntPath("ITER", 1 - pyas.odessa_shift(), SEQUENCE)
LOOKUNIT = flpa.IntPath("LOOKUNIT", 1 - pyas.odessa_shift(), SEQUENCE)
CPUTIME = flpa.FloatPath("CPUTIME", 1 - pyas.odessa_shift(), SEQUENCE)

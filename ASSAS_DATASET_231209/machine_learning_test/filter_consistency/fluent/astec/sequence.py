import fluent.path as flpa
import fluent.astec as flas

SEQUENCE = flpa.BasePath("SEQUENCE", 0, flas.ROOT)
TSCRAM = flpa.FloatPath("TSCRAM", 0, SEQUENCE)
STEP = flpa.FloatPath("STEP", 0, SEQUENCE)

TIME = flpa.FloatPath("TIME", 0, SEQUENCE)
TIMA = flpa.FloatPath("TIMA", 0, SEQUENCE)
ITER0 = flpa.IntPath("ITER0", 0, SEQUENCE)
ITER = flpa.IntPath("ITER", 0, SEQUENCE)
LOOKUNIT = flpa.IntPath("LOOKUNIT", 0, SEQUENCE)
CPUTIME = flpa.FloatPath("CPUTIME", 0, SEQUENCE)

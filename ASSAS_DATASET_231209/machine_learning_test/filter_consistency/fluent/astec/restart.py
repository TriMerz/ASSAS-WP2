import fluent.path as flpa
import fluent.astec as flas


RESTART = flpa.BasePath("RESTART", 0, flas.ROOT)
SEQUENCE = flpa.BasePath("SEQUENCE", 0, RESTART)
SEQUENCE_TIME = flpa.FloatPath("TIME", 0, SEQUENCE)
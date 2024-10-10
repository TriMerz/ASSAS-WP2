import fluent.path as flpa
import fluent.astec as flas
import pyastec as pyas


RESTART = flpa.BasePath("RESTART", 1 - pyas.odessa_shift(), flas.ROOT)
SEQUENCE = flpa.BasePath("SEQUENCE", 1 - pyas.odessa_shift(), RESTART)
SEQUENCE_TIME = flpa.FloatPath("TIME", 1 - pyas.odessa_shift(), SEQUENCE)
import fluent.path as flpa
import fluent.astec as flas
import pyastec as pyas

VESSEL = flpa.BasePath("VESSEL", 1 - pyas.odessa_shift(), flas.ROOT)

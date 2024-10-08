import fluent.path as flpa
import fluent.astec as flas

def sensor(index:int):
    return flpa.BasePath("SENSOR", index, flas.ROOT)

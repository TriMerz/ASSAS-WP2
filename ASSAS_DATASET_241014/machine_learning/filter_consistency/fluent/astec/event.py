import fluent.path as flpa
import fluent.astec as flas

def event(index:int):
    return flpa.BasePath("EVENT", index, flas.ROOT)

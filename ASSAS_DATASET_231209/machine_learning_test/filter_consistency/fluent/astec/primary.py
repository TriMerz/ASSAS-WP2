import fluent.path as flpa
import fluent.astec as flas

PRIMARY = flpa.BasePath("PRIMARY", 0, flas.ROOT)

def junction(pos:int) -> flpa.BasePath:
    return flpa.BasePath("JUNCTION", pos, PRIMARY)

def name(junction:flpa.BasePath) -> flpa.StrPath:
    return flpa.StrPath("NAME", 0, junction)

def nv_up(junction:flpa.BasePath) -> flpa.StrPath:
    return flpa.StrPath("NV_UP", 0, junction)

def nv_down(junction:flpa.BasePath) -> flpa.StrPath:
    return flpa.StrPath("NV_DOWN", 0, junction)

def primary_junction_ther(junction:flpa.BasePath, pos:int):
    return flpa.BasePath("THER", pos, junction)
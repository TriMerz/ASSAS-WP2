import fluent.path as flpa
import fluent.astec as flas

def connecti(index:int):
    return flpa.BasePath("CONNECTI", index, flas.ROOT)

def connecti_name(index:int):
    return flpa.StrPath("NAME", 0, connecti(index))

def connecti_from(index:int):
    return flpa.StrPath("FROM", 0, connecti(index))

def connecti_to(index:int):
    return flpa.StrPath("TO", 0, connecti(index))

def STAT(connecti:flpa.BasePath):
    return flpa.OdtPath("STAT", 0, connecti)

def H(connecti:flpa.BasePath, index:int):
    return flpa.OdtPath("H", index, connecti)


def T(connecti:flpa.BasePath, index:int):
    return flpa.OdtPath("T", index, connecti)
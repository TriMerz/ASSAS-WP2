import fluent.path as flpa
import fluent.astec as flas
import pyastec as pyas

def connecti(index:int):
    return flpa.BasePath("CONNECTI", index, flas.ROOT)

def connecti_name(index:int):
    return flpa.StrPath("NAME", 1 - pyas.odessa_shift(), connecti(index))

def connecti_from(index:int):
    return flpa.StrPath("FROM", 1 - pyas.odessa_shift(), connecti(index))

def connecti_to(index:int):
    return flpa.StrPath("TO", 1 - pyas.odessa_shift(), connecti(index))

def STAT(connecti:flpa.BasePath):
    return flpa.OdtPath("STAT", 1 - pyas.odessa_shift(), connecti)

def H(connecti:flpa.BasePath, index:int):
    return flpa.OdtPath("H", index, connecti)


def T(connecti:flpa.BasePath, index:int):
    return flpa.OdtPath("T", index, connecti)
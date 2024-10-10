import fluent.path as pa
import fluent.fluent_astec as flas
import pyastec as pyas

class CalcOpt(pa.BaseFamilyPath):
    def __init__(self, parent:flas.AstecRoot=flas.AstecRoot()) -> None:
        super().__init__("CALC_OPT", 1 - pyas.odessa_shift(), parent)

    def cesar(self):
        return Cesar(self)

class Cesar(pa.BaseFamilyPath):
    def __init__(self, parent:CalcOpt=CalcOpt()) -> None:
        super().__init__("CESAR", 1 - pyas.odessa_shift(), parent)

    def fatcl_dl(self):
        return CesarFactl_Dl(self)
    
    def stop_flux(self):
        return CesarStopFlux(self)
    

class CesarFactl_Dl(pa.FloatFamilyPath):
    def __init__(self, parent:Cesar=Cesar()) -> None:
        super().__init__("factL_dL", 1 - pyas.odessa_shift(), parent)

class CesarStopFlux(pa.FloatFamilyPath):
    def __init__(self, parent:Cesar=Cesar()) -> None:
        super().__init__("StopFlux", 1 - pyas.odessa_shift(), parent)
import fluent.path as pa
import pyastec as pyas

class AstecRoot(pa.Root):

    def __init__(self) -> None:
        super().__init__()

    def loadtime(self):
        return LoadTime(self)
    
    def visu(self):
        return Visu(self)
    
    def restart(self):
        return Restart(self)
    
    def save(self):
        return Save(self)
    
    def warnings(self):
        return Warnings(self)

    def calc_opt(self):
        from fluent.fluent_astec.calcopt import CalcOpt
        return CalcOpt(self)
    
    def sequence(self):
        from fluent.fluent_astec.sequence import Sequence  
        return Sequence(self)
    
    def containm(self):
        from fluent.fluent_astec.containm import Containm  
        return Containm(self)
        
    def primary(self):
        from fluent.fluent_astec.primary import Primary  
        return Primary(self)
    
    def secondar(self):
        from fluent.fluent_astec.secondar import Secondar  
        return Secondar(self)
    
    def vessel(self):
        from fluent.fluent_astec.vessel import Vessel  
        return Vessel(self)
    
    def connecti(self, pos:int):
        from fluent.fluent_astec.connecti import Connecti  
        return Connecti(pos, self)
    
    def systems(self):
        from fluent.fluent_astec.systems import Systems  
        return Systems(self)
    
    def event(self, pos:int):
        from fluent.fluent_astec.event import Event  
        return Event(pos, self)
    
    def sensor(self, pos:int):
        from fluent.fluent_astec.sensor import Sensor  
        return Sensor(pos, self)
    
    
ROOT = AstecRoot()

class LoadTime(pa.FamilyPath):
    def __init__(self, parent:AstecRoot=AstecRoot()) -> None:
        super().__init__(pyas.od_r0, "LOADTIME", 1 - pyas.odessa_shift(), parent)

class Visu(pa.BaseFamilyPath):
    def __init__(self, parent:AstecRoot=AstecRoot()) -> None:
        super().__init__("VISU", 1 - pyas.odessa_shift(), parent)

class Restart(pa.BaseFamilyPath):
    def __init__(self, parent:AstecRoot=AstecRoot()) -> None:
        super().__init__("RESTART", 1 - pyas.odessa_shift(), parent)

class Save(pa.BaseFamilyPath):
    def __init__(self, parent:AstecRoot=AstecRoot()) -> None:
        super().__init__("SAVE", 1 - pyas.odessa_shift(), parent)

class Warnings(pa.FamilyPath[int]):
    def __init__(self, parent:AstecRoot=AstecRoot()) -> None:
        super().__init__(pyas.od_i0, "WARNINGS", 1 - pyas.odessa_shift(), parent)


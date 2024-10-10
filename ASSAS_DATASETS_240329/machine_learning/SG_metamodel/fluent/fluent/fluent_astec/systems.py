import fluent.path as pa
import fluent.fluent_astec as flas
from fluent.path import BasePath
import pyastec as pyas

class Systems(pa.BaseFamilyPath):
    def __init__(self, parent: flas.AstecRoot=flas.AstecRoot()) -> None:
        super().__init__("SYSTEMS", 1 - pyas.odessa_shift(), parent)

    def accumula(self, pos:int):
        return Accumula(pos, self)
    
    def pump(self, pos:int):
        return Pump(pos, self)
        
    def tank(self, pos:int):
        return Tank(pos, self)

class Accumula(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Systems) -> None:
        super().__init__("ACCUMULA", pos, parent)

class Pump(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Systems) -> None:
        super().__init__("PUMP", pos, parent)

class Tank(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Systems) -> None:
        super().__init__("TANK", pos, parent)


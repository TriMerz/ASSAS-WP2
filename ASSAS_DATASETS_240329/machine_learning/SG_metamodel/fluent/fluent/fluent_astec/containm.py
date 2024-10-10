import fluent.path as pa
import fluent.fluent_astec as flas
import pyastec as pyas

class Containm(pa.BaseFamilyPath):
    def __init__(self, parent: flas.AstecRoot = flas.AstecRoot()) -> None:
        super().__init__("CONTAINM", 1 - pyas.odessa_shift(), parent)

    def wall(self, pos:int):
        return Wall(pos, self)
    
    def conn(self, pos:int):
        return Conn(pos, self)

class Conn(pa.BaseFamilyPath):
    def __init__(self,  pos: int, parent: Containm=Containm()) -> None:
        super().__init__("CONN", pos, parent)

    def to(self):
        return ConnTo(self)
    
    def from_(self):
        return ConnFrom(self)
    
    def name_(self):
        return ConnName(self)
    
    def pipe(self, pos:int):
        return Pipe(pos, self)
    

class ConnTo(pa.TextFamilyPath):
    def __init__(self, parent: Conn) -> None:
        super().__init__("TO", 1 - pyas.odessa_shift(), parent)

class ConnFrom(pa.TextFamilyPath):
    def __init__(self, parent: Conn) -> None:
        super().__init__("FROM", 1 - pyas.odessa_shift(), parent)

class ConnName(pa.StrFamilyPath):
    def __init__(self, parent: Conn) -> None:
        super().__init__("NAME", 1 - pyas.odessa_shift(), parent)

class Pipe(pa.BaseFamilyPath):
    def __init__(self,  pos: int, parent: Conn) -> None:
        super().__init__("PIPE", pos, parent)

    def from_int(self):
        return PipeFromInt(self)
    
    def from_str(self):
        return PipeFromStr(self)
    
    def to_int(self):
        return PipeToInt(self)
    
    def to_str(self):
        return PipeToStr(self)
    
    def coms(self):
        return PipeComs(self)

class PipeFromInt(pa.IntFamilyPath):
    def __init__(self, parent: Pipe) -> None:
        super().__init__("FROM", 1 - pyas.odessa_shift(), parent)

class PipeFromStr(pa.StrFamilyPath):
    def __init__(self, parent: Pipe) -> None:
        super().__init__("FROM", 1 - pyas.odessa_shift(), parent)

class PipeToInt(pa.IntFamilyPath):
    def __init__(self, parent: Pipe) -> None:
        super().__init__("TO", 1 - pyas.odessa_shift(), parent)

class PipeToStr(pa.StrFamilyPath):
    def __init__(self,  parent: Pipe) -> None:
        super().__init__("TO", 1 - pyas.odessa_shift(), parent)

class PipeComs(pa.TextFamilyPath):
    def __init__(self, parent: Pipe) -> None:
        super().__init__("COMS", 1 - pyas.odessa_shift(), parent)
        
class Wall(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Containm=Containm()) -> None:
        super().__init__("WALL", pos, parent)

    def name_(self):
        return WallName(self)
    
    def heat(self, pos:int):
        return WallHeat(pos, self)

class WallName(pa.StrFamilyPath):
    def __init__(self, parent: Wall) -> None:
        super().__init__("NAME", 1 - pyas.odessa_shift(), parent)

class WallHeat(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Wall) -> None:
        super().__init__("HEAT", pos, parent)

    def medi(self):
        return WallHeatMedi(self)

class WallHeatMedi(pa.TextFamilyPath):
    def __init__(self, parent: WallHeat) -> None:
        super().__init__("MEDI", 1 - pyas.odessa_shift(), parent)
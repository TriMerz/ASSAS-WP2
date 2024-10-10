import fluent.path as pa
import fluent.fluent_astec as flas
from fluent.path import BasePath
import pyastec as pyas

class Secondar(pa.BaseFamilyPath):
    def __init__(self, parent:flas.AstecRoot = flas.AstecRoot()) -> None:
        super().__init__("SECONDAR", 1 - pyas.odessa_shift(), parent)

    def wall(self, pos:int):
        return Wall(pos, self)
    
    def junction(self, pos:int):
        return Junction(pos, self)
    
    def volume(self, pos:int):
        return Volume(pos, self)
    
class Volume(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Secondar) -> None:
        super().__init__("VOLUME", pos, parent)

    def ther(self):
        return VolumeTher(self)

class VolumeTher(pa.BaseFamilyPath):
    def __init__(self, parent:Volume) -> None:
        super().__init__("THER", 1 - pyas.odessa_shift(), parent)

class Wall(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent:Secondar = Secondar()) -> None:
        super().__init__("WALL", pos, parent)

    def volume(self):
        return WallVolume(self)
    
    def name_(self):
        return WallName(self)

    def ther(self, pos:int):
        return WallTher(pos, self)

class WallVolume(pa.StrFamilyPath):
    def __init__(self, parent:Wall) -> None:
        super().__init__("VOLUME", 1 - pyas.odessa_shift(), parent)

class WallName(pa.StrFamilyPath):
    def __init__(self, wall:Wall) -> None:
        super().__init__("NAME", 1 - pyas.odessa_shift(), wall)

class WallTher(pa.BaseFamilyPath):
    def __init__(self, pos:int, wall:Wall) -> None:
        super().__init__("THER", pos, wall)

class Junction(pa.BaseFamilyPath):
    def __init__(self, pos:int, parent:Secondar=Secondar()) -> None:
        super().__init__("JUNCTION", pos, parent)

    def name_(self):
        return JunctionName(self)
    
    def close(self):
        return JunctionClose(self)
    
    def nv_up(self):
        return JunctionNvUp(self)
    
    def nv_down(self):
        return JunctionNvDown(self)
    
    def ther(self, pos:int):
        return JunctionTher(pos, self)

class JunctionName(pa.StrFamilyPath):
    def __init__(self, parent: Junction) -> None:
        super().__init__("NAME", 1 - pyas.odessa_shift(), parent)

class JunctionClose(pa.IntFamilyPath):
    def __init__(self, parent: Junction) -> None:
        super().__init__("CLOSE", 1 - pyas.odessa_shift(), parent)

class JunctionNvUp(pa.StrFamilyPath):
    def __init__(self, parent: Junction) -> None:
        super().__init__("NV_UP", 1 - pyas.odessa_shift(), parent)

class JunctionNvDown(pa.StrFamilyPath):
    def __init__(self, parent: Junction) -> None:
        super().__init__("NV_DOWN", 1 - pyas.odessa_shift(), parent)

class JunctionTher(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Junction) -> None:
        super().__init__("THER", pos, parent)

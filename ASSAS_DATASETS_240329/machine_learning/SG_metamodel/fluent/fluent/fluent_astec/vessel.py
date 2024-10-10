import fluent.path as pa
import fluent.fluent_astec as flas
import pyastec as pyas

class Vessel(pa.BaseFamilyPath):
    def __init__(self, parent: flas.AstecRoot=flas.AstecRoot()) -> None:
        super().__init__("VESSEL", 1 - pyas.odessa_shift(), parent)

    def mesh(self, pos:int):
        return Mesh(pos, self)
    
    def face(self, pos:int):
        return Face(pos, self)
    
    def comp(self, pos:int):
        return Comp(pos, self)
    
    def cond(self, pos:int):
        return Cond(pos, self)
    
    def macr(self, pos:int):
        return Macr(pos, self)

class Mesh(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Vessel) -> None:
        super().__init__("MESH", pos, parent)

    def name_(self):
        return MeshName(self)

class MeshName(pa.StrFamilyPath):
    def __init__(self, parent: Mesh) -> None:
        super().__init__("NAME", 1 - pyas.odessa_shift(), parent)

class Face(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Vessel) -> None:
        super().__init__("FACE", pos, parent)

    def name_(self):
        return FaceName(self)
    
    def type(self):
        return FaceType(self)
    
    def first_mesh_index(self):
        return FaceFirstMeshIndex(self)
    
    def second_mesh_index(self):
        return FaceSecondMeshIndex(self)

class FaceName(pa.StrFamilyPath):
    def __init__(self, parent: Face) -> None:
        super().__init__("NAME", 1 - pyas.odessa_shift(), parent)
        
class FaceType(pa.StrFamilyPath):
    def __init__(self, parent: Face) -> None:
        super().__init__("TYPE", 1 - pyas.odessa_shift(), parent)

class FaceFirstMeshIndex(pa.IntFamilyPath):
    def __init__(self, parent: Face) -> None:
        super().__init__("MESH", 1 - pyas.odessa_shift(), parent)

class FaceSecondMeshIndex(pa.IntFamilyPath):
    def __init__(self, parent: Face) -> None:
        super().__init__("MESH", 1 - pyas.odessa_shift(), parent)

class Comp(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Vessel = Vessel()) -> None:
        super().__init__("COMP", pos, parent)

    def mesh(self):
        return CompMesh(self)
    
    def stat(self):
        return CompStat(self)
    
    def macr(self):
        return CompMacr(self)

class CompMesh(pa.IntFamilyPath):
    def __init__(self, parent: Comp) -> None:
        super().__init__("MESH", 1 - pyas.odessa_shift(), parent)

class CompStat(pa.StrFamilyPath):
    def __init__(self, parent: Comp) -> None:
        super().__init__("STAT", 1 - pyas.odessa_shift(), parent)

class CompMacr(pa.StrFamilyPath):
    def __init__(self, parent: Comp) -> None:
        super().__init__("MACR", 1 - pyas.odessa_shift(), parent)

class Macr(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Vessel = Vessel()) -> None:
        super().__init__("MACR", pos, parent)

    def name_(self):
        return MacrName(self)
    
    def inside(self):
        return MacrInside(self)

class MacrName(pa.StrFamilyPath):
    def __init__(self, parent: Macr) -> None:
        super().__init__("NAME", 1 - pyas.odessa_shift(), parent)

class MacrInside(pa.StrFamilyPath):
    def __init__(self, parent: Macr) -> None:
        super().__init__("INSIDE", 1 - pyas.odessa_shift(), parent)

class Cond(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: Vessel = Vessel()) -> None:
        super().__init__("COND", pos, parent)

    def macr(self, pos:int):
        return CondMacr(pos, self)

class CondMacr(pa.StrFamilyPath):
    def __init__(self, pos: int, parent: Cond) -> None:
        super().__init__("MACE", pos, parent)
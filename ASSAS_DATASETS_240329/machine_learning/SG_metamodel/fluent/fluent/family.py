import pyastec as pyas
import typing
import abc
import fluent.cache as cache
import fluent.path as pa
import fluent.fluent_odessa as flod

class Family(abc.ABC, typing.Generic[flod.T]):

    def __init__(self, typ:int, name:str, parent:pa.BasePath) -> None:
        self.typ = typ
        self.name = name
        self.parent = parent

    @abc.abstractmethod
    def path(self, pos:int) -> pa.FamilyPath:
        pass
    
class BaseFamily(Family[pyas.odbase]):
    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_base, name, parent)

    def path(self, pos: int):
        return pa.BaseFamilyPath(self.name, pos, self.parent)
    
    def name_index_from(self, name:str, base:cache.CachedOptOdbase):
        return pyas.odbase_locate(self.parent.get_from(base), self.name, name)


class TextFamily(Family[pyas.odt]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_t, name, parent)

    def path(self, pos: int):
        return pa.TextFamilyPath(self.name, pos, self.parent)
    
class StrFamily(Family[str]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_c0, name, parent)

    def path(self, pos: int):
        return pa.StrFamilyPath(self.name, pos, self.parent)
    
class FloatFamily(Family[float]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_r0, name, parent)

    def path(self, pos: int):
        return pa.FloatFamilyPath(self.name, pos, self.parent)
    
class IntFamily(Family[int]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_i0, name, parent)

    def path(self, pos: int):
        return pa.IntFamilyPath(self.name, pos, self.parent)


class I1Family(Family[pyas.odi1]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_i1, name, parent)

    def path(self, pos: int):
        return pa.I1FamilyPath(self.name, pos, self.parent)
    
class I2Family(Family[pyas.odi2]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_i2, name, parent)

    def path(self, pos: int):
        return pa.I2FamilyPath(self.name, pos, self.parent)
    
class R1Family(Family[pyas.odr1]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_r1, name, parent)

    def path(self, pos: int):
        return pa.R1FamilyPath(self.name, pos, self.parent)
    
class R2Family(Family[pyas.odr2]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_r2, name, parent)

    def path(self, pos: int):
        return pa.R2FamilyPath(self.name, pos, self.parent)
    
class R3Family(Family[pyas.odr3]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_r3, name, parent)

    def path(self, pos: int):
        return pa.R3FamilyPath(self.name, pos, self.parent)
    
class C1Family(Family[pyas.odc1]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_c1, name, parent)

    def path(self, pos: int):
        return pa.C1FamilyPath(self.name, pos, self.parent)
    

class IGFamily(Family[pyas.odig]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_ig, name, parent)

    def path(self, pos: int):
        return pa.IGFamilyPath(self.name, pos, self.parent)

class RGFamily(Family[pyas.odrg]):

    def __init__(self, name: str, parent: pa.BasePath) -> None:
        super().__init__(pyas.od_rg, name, parent)

    def path(self, pos: int):
        return pa.RGFamilyPath(self.name, pos, self.parent)

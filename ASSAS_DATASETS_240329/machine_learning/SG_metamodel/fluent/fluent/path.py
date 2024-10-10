import abc
import typing
import pyastec as pyas
import fluent.cache as cache
import fluent.fluent_odessa as flod

class Path(abc.ABC, typing.Generic[flod.T]):

    def __init__(self) -> None:
        super().__init__()
        self.odessa_repr = self._odessa_repr()

    def __str__(self) -> str:
        return self.odessa_repr

    def __repr__(self):
        return self.__str__()
    
    @abc.abstractmethod
    def _odessa_repr(self) -> str:
        pass
    
    @abc.abstractmethod
    def get_from(self, base:cache.CachedOptOdbase) -> flod.T:
        pass

    @abc.abstractmethod
    def exists_from(self, base:cache.CachedOptOdbase) -> bool:
        pass

class BasePath(Path[pyas.odbase]):

    def size_from(self, name:str, base:cache.CachedOptOdbase):
        return pyas.odbase_size(self.get_from(base), name)

    def child_float(self, name:str, pos:int):
        return FloatFamilyPath(name, pos, self)
    
    def child_int(self, name:str, pos:int):
        return IntFamilyPath(name, pos, self)
    
    def child_str(self, name:str, pos:int):
        return StrFamilyPath(name, pos, self)
    
    def child_base(self, name:str, pos:int):
        return BaseFamilyPath(name, pos, self)
    
    def child_text(self, name:str, pos:int):
        return TextFamilyPath(name, pos, self)
    
    def child_i1(self, name:str, pos:int):
        return I1FamilyPath(name, pos, self)
    
    def child_i2(self, name:str, pos:int):
        return I2FamilyPath(name, pos, self)
    
    def child_r1(self, name:str, pos:int):
        return R1FamilyPath(name, pos, self)
    
    def child_r2(self, name:str, pos:int):
        return R2FamilyPath(name, pos, self)
    
    def child_r3(self, name:str, pos:int):
        return R3FamilyPath(name, pos, self)
    
    def child_c1(self, name:str, pos:int):
        return C1FamilyPath(name, pos, self)
    
    def child_name(self):
        return self.child_str("NAME", 1 - pyas.odessa_shift())
    
    def name_from(self, base:cache.CachedOptOdbase):
        return self.child_name().get_from(base)

class Root(BasePath):

    def __init__(self) -> None:
        super().__init__()

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Root)

    def get_from(self, base: cache.CachedOptOdbase):
        if isinstance(base, pyas.odbase):
            return base
        else:
            return base.base
        
    def exists_from(self, base: cache.CachedOptOdbase) -> bool:
        return True
    
    def _odessa_repr(self) -> str:
        return "BASE"
    
class FamilyPath(Path[flod.T]):
    
    setters_put = {
        pyas.od_base: pyas.odbase_put_odbase,
        pyas.od_t:pyas.odbase_put_odt,
        pyas.od_c0:pyas.odbase_put_string,
        pyas.od_c1:pyas.odbase_put_odc1,
        pyas.od_i0:pyas.odbase_put_int,
        pyas.od_i1:pyas.odbase_put_odi1,
        pyas.od_i2:pyas.odbase_put_odi2,
        pyas.od_ig:pyas.odbase_put_odig,
        pyas.od_r0:pyas.odbase_put_double,
        pyas.od_r1:pyas.odbase_put_odr1,
        pyas.od_r2:pyas.odbase_put_odr2,
        pyas.od_r3:pyas.odbase_put_odr3,
        pyas.od_rg:pyas.odbase_put_odrg,
    }

    setters_insert = {
        pyas.od_base:pyas.odbase_insert_odbase,
        pyas.od_t:pyas.odbase_insert_odt,
        pyas.od_c0:pyas.odbase_insert_string,
        pyas.od_c1:pyas.odbase_insert_odc1,
        pyas.od_i0:pyas.odbase_insert_int,
        pyas.od_i1:pyas.odbase_insert_odi1,
        pyas.od_i2:pyas.odbase_insert_odi2,
        pyas.od_ig:pyas.odbase_insert_odig,
        pyas.od_r0:pyas.odbase_insert_double,
        pyas.od_r1:pyas.odbase_insert_odr1,
        pyas.od_r2:pyas.odbase_insert_odr2,
        pyas.od_r3:pyas.odbase_insert_odr3,
        pyas.od_rg:pyas.odbase_insert_odrg,
    }

    getters = {
        pyas.od_base: pyas.odbase_get_odbase,
        pyas.od_t:pyas.odbase_get_odt,
        pyas.od_c0:pyas.odbase_get_string,
        pyas.od_c1:pyas.odbase_get_odc1,
        pyas.od_i0:pyas.odbase_get_int,
        pyas.od_i1:pyas.odbase_get_odi1,
        pyas.od_i2:pyas.odbase_get_odi2,
        pyas.od_ig:pyas.odbase_get_odig,
        pyas.od_r0:pyas.odbase_get_double,
        pyas.od_r1:pyas.odbase_get_odr1,
        pyas.od_r2:pyas.odbase_get_odr2,
        pyas.od_r3:pyas.odbase_get_odr3,
        pyas.od_rg:pyas.odbase_get_odrg,
    }

    def __init__(self, typ:int, name:str, pos:int, parent:BasePath) -> None:
        self.typ = typ
        self.parent = parent
        self.name = name
        self.pos = pos
        super().__init__()

    def _odessa_repr(self) -> str:
        return f"{self.parent}:{self.name}[{self.pos}]"

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, FamilyPath) \
                and self.name == __value.name \
                    and self.pos == __value.pos \
                        and self.parent == __value.parent
    
    def get_from(self, base:cache.CachedOptOdbase):
        if isinstance(base, pyas.odbase):
            inline_base = self.parent.get_from(base)
            return self.get_from_inline(inline_base)
        else:
            if self.odessa_repr in base.cache:
                return base.cache[self.odessa_repr]
            else:
                inline_base = self.parent.get_from(base)
                result = self.get_from_inline(inline_base)
                base.cache[self.odessa_repr] = result
                return result
        
    def get_from_inline(self, base:pyas.odbase) -> flod.T:
        if self.typ in self.getters:
            return self.getters[self.typ](base, self.name, self.pos)
        else:
            raise Exception(f"Unhandled typ : {flod.typ_names[self.typ]}")
            
    def exists_from(self, base: cache.CachedOptOdbase) -> bool:
        if self.parent.exists_from(base):
            parent = self.parent.get_from(base)
            if pyas.odbase_type(parent, self.name) == self.typ:
                return self.pos < pyas.odbase_size(parent, self.name) + 1 - pyas.odessa_shift()
            else:
                return False
        else:
            return False

    def put_from(self, base:cache.CachedOptOdbase, value:flod.T):
        local_base = self.parent.get_from(base)
        if self.typ in self.setters_put:
            self.setters_put[self.typ](local_base, self.name, value, self.pos)
        else:
            raise Exception(f"Unhandled type : {flod.typ_names[self.typ]} for {self}")

    def insert_from(self, base:cache.CachedOptOdbase, value:flod.T):
        local_base = self.parent.get_from(base)
        if self.typ in self.setters_insert:
            self.setters_insert[self.typ](local_base, self.name, value, self.pos)
        else:
            raise Exception(f"Unhandled type : {flod.typ_names[self.typ]} for {self}")


class BaseFamilyPath(FamilyPath[pyas.odbase], BasePath):

    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_base, name, pos, parent)
    
class TextFamilyPath(FamilyPath[pyas.odt]):

    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_t, name, pos, parent)
    
    def get_from_as_str(self, base:cache.CachedOptOdbase)->str:
        return pyas.odt_get(self.get_from(base))

class StrFamilyPath(FamilyPath[str]):

    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_c0, name, pos, parent)

class FloatFamilyPath(FamilyPath[float]):

    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_r0, name, pos, parent)

class IntFamilyPath(FamilyPath[int]):

    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_i0, name, pos, parent)

class ArrayPath(FamilyPath[flod.AT]):
    pass

class I1FamilyPath(ArrayPath[pyas.odi1]):
    
    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_i1, name, pos, parent)

    def child_element(self, index:int):
        return I1ElementPath((index,), self)

class I2FamilyPath(ArrayPath[pyas.odi2]):
    
    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_i2, name, pos, parent)
    
    def child_element(self, index1:int, index2:int):
        return I2ElementPath((index1, index2), self)

class R1FamilyPath(ArrayPath[pyas.odr1]):
    
    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_r1, name, pos, parent)

    def child_element(self, index:int):
        return R1ElementPath((index,), self)
    
class R2FamilyPath(ArrayPath[pyas.odr2]):
    
    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_r2, name, pos, parent)
    
    def child_element(self, index1:int, index2:int):
        return R2ElementPath((index1, index2), self)
    

class R3FamilyPath(ArrayPath[pyas.odr3]):
    
    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_r3, name, pos, parent)
    
    
    def child_element(self, index1:int, index2:int, index3:int):
        return R3ElementPath((index1, index2, index3), self)

class C1FamilyPath(ArrayPath[pyas.odc1]):
    
    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_c1, name, pos, parent)
    
    def child_element(self, index:int):
        return C1ElementPath((index,), self)

class ArrayElementPath(typing.Generic[flod.X, flod.AT, flod.C], Path[flod.X]):

    def __init__(self, dim:int, coord:flod.C, parent: ArrayPath[flod.AT]) -> None:
        self.coord = coord
        self.dim = dim
        self.parent = parent
        super().__init__()

    def __str__coord__(self):
        return ','.join(map(str, self.coord))

    def _odessa_repr(self) -> str:
        return f"{self.parent}[{self.__str__coord__()}]"

    def get_from(self, base: cache.CachedOptOdbase):
        return self.get_from_inline(self.parent.get_from(base))
    
    def get_from_inline(self, array:flod.AT) -> flod.X:
        if isinstance(self, I1ElementPath):
            return pyas.odi1_get(array, self.coord[0])
        elif isinstance(self, I2ElementPath):
            return pyas.odi2_get(array, self.coord[0], self.coord[1])
        elif isinstance(self, R1ElementPath):
            return pyas.odr1_get(array, self.coord[0])
        elif isinstance(self, R2ElementPath):
            return pyas.odr2_get(array, self.coord[0], self.coord[1])
        elif isinstance(self, R3ElementPath):
            return pyas.odr3_get(array, self.coord[0], self.coord[1], self.coord[2])
        elif isinstance(self, C1ElementPath):
            return pyas.odc1_get(array, self.coord[0])
        else:
            raise Exception(f"Unhandled group path : {self}")
    
    def exists_from(self, base: cache.CachedOptOdbase) -> bool:
        if self.parent.exists_from(base):
            parent = self.parent.get_from(base)
            if isinstance(self, I1ElementPath):
                return self.coord[0] < pyas.odi1_size(parent) + 1 - pyas.odessa_shift()
            elif isinstance(self, I2ElementPath):
                return self.coord[0] < pyas.odi2_card(parent, 0) + 1 - pyas.odessa_shift() \
                    and self.coord[1] < pyas.odi2_card(parent, 1) + 1 - pyas.odessa_shift()
            elif isinstance(self, R1ElementPath):
                return self.coord[0] < pyas.odr1_size(parent) + 1 - pyas.odessa_shift()
            elif isinstance(self, R2ElementPath):
                return self.coord[0] < pyas.odr2_card(parent, 0) + 1 - pyas.odessa_shift() \
                    and self.coord[1] < pyas.odr2_card(parent, 1) + 1 - pyas.odessa_shift()
            elif isinstance(self, R3ElementPath):
                return self.coord[0] < pyas.odr3_card(parent, 0) + 1 - pyas.odessa_shift() \
                    and self.coord[1] < pyas.odr3_card(parent, 1) + 1 - pyas.odessa_shift() \
                    and self.coord[2] < pyas.odr3_card(parent, 2) + 1 - pyas.odessa_shift()
            elif isinstance(self, C1ElementPath):
                return self.coord[0] < pyas.odc1_size(parent) + 1 - pyas.odessa_shift()
            else:
                raise Exception(f"Unhandled array path {self}")
        else:
            return False
    
class I1ElementPath(ArrayElementPath[int, pyas.odi1, flod.C_1D]):
    def __init__(self, coord: flod.C_1D, parent: ArrayPath[pyas.odi1]) -> None:
        super().__init__(1, coord, parent)

class I2ElementPath(ArrayElementPath[int, pyas.odi2, flod.C_2D]):
    def __init__(self, coord: flod.C_2D, parent: ArrayPath[pyas.odi2]) -> None:
        super().__init__(2, coord, parent)

class R1ElementPath(ArrayElementPath[int, pyas.odr1, flod.C_1D]):
    def __init__(self, coord: flod.C_1D, parent: ArrayPath[pyas.odr1]) -> None:
        super().__init__(1, coord, parent)

class R2ElementPath(ArrayElementPath[int, pyas.odr2, flod.C_2D]):
    def __init__(self, coord: flod.C_2D, parent: ArrayPath[pyas.odr2]) -> None:
        super().__init__(2, coord, parent)

class R3ElementPath(ArrayElementPath[int, pyas.odr3, flod.C_3D]):
    def __init__(self, coord: flod.C_3D, parent: ArrayPath[pyas.odr3]) -> None:
        super().__init__(3, coord, parent)

class C1ElementPath(ArrayElementPath[int, pyas.odc1, flod.C_1D]):
    def __init__(self, coord: flod.C_1D, parent: ArrayPath[pyas.odc1]) -> None:
        super().__init__(1, coord, parent)

class GroupPath(FamilyPath[flod.GT]):
    pass

class IGFamilyPath(GroupPath[pyas.odig]):
    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_ig, name, pos, parent)

class RGFamilyPath(GroupPath[pyas.odrg]):
    def __init__(self, name: str, pos: int, parent: BasePath) -> None:
        super().__init__(pyas.od_rg, name, pos, parent)

class GroupElementPath(typing.Generic[flod.X, flod.GT], Path[flod.X]):
    def __init__(self, name:str, parent: FamilyPath[flod.GT]) -> None:
        self.name = name
        self.parent = parent
        super().__init__()
    
    def _odessa_repr(self) -> str:
        return f"{self.parent}[{self.name}]"

    def get_from(self, base: cache.CachedOptOdbase):
        return self.get_from_inline(self.parent.get_from(base))
    
    def get_from_inline(self, group:flod.GT) -> flod.X:
        if isinstance(self, IGElementPath):
            return pyas.odig_get(group, self.name)
        elif isinstance(self, RGElementPath):
            return pyas.odrg_get(group, self.name)
        else:
            raise Exception(f"Unhandled group path : {self}")
    
    def exists_from(self, base: cache.CachedOptOdbase) -> bool:
        if self.parent.exists_from(base):
            return True # TODO add function to pyastec to check if name exists in group
        else:
            return False
    
class IGElementPath(GroupElementPath[int, pyas.odig]):

    def __init__(self, name: str, parent: FamilyPath[pyas.odig]) -> None:
        super().__init__(name, parent)

class RGElementPath(GroupElementPath[float, pyas.odrg]):

    def __init__(self, name: str, parent: FamilyPath[pyas.odrg]) -> None:
        super().__init__(name, parent)        


    

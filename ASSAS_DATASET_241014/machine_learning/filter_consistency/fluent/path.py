from __future__ import annotations
import abc
import typing
import swigodessa as so
import fluent.types as flty
import itertools
import logging

C_1D = typing.Tuple[int]
C_2D = typing.Tuple[int, int]
C_3D = typing.Tuple[int, int, int]
T = typing.TypeVar('T', so.odbase, float, so.odr1, so.odr2, so.odr3, int, so.odi1, so.odi2, str, so.odc1, so.odrg, so.odig, so.odelem, so.odt)
X = typing.TypeVar('X', float, int, str)
C = typing.TypeVar('C', C_1D, C_2D, C_3D)

class Path(abc.ABC, typing.Generic[T]):
    
    def __init__(self, typ:int) -> None:
        self.typ = typ

    @abc.abstractmethod
    def __getitem__(self, base:so.odbase) -> T:
        pass
        
    @abc.abstractmethod
    def __setitem__(self, base:so.odbase, value:T) -> None:
        pass

    @abc.abstractmethod
    def check(self, base:so.odbase) -> bool:
        pass

    def __str__(self) -> str:
        return "BASE"

    def __repr__(self):
        return self.__str__()

class Root(Path[so.odbase]):

    def __init__(self) -> None:
        super().__init__(so.od_base)

    def __getitem__(self, base: so.odbase) -> so.odbase:
        return base

    def __setitem__(self, base: so.odbase, value:so.odbase) -> None:
        for path in enumerate_base(Root(), value):
            path[base] = path[value]

    def check(self, base: so.odbase) -> bool:
        return True

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Root)

class FamilyPath(Path[T]):
    
    def __init__(self, typ:int, name:str, pos:int, parent:Path[so.odbase]) -> None:
        super().__init__(typ)
        self.parent = parent
        self.name = name
        self.pos = pos

    def __getitem__(self, base:so.odbase) -> T:
        return self.get_in_base(self.parent[base])

    def check(self, base:so.odbase) -> bool:
        self.parent.check(base)
        base = self.parent[base]
        return self.pos < so.odbase_size(base, self.name)
    
    @abc.abstractmethod
    def get_in_base(self, base:so.odbase) -> T:
        pass

    def __str__(self) -> str:
        return f"{self.parent}:{self.name}[{self.pos}]"

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, FamilyPath) \
                and self.name == __value.name \
                    and self.pos == __value.pos \
                        and self.parent == __value.parent

SizeCallable = typing.Callable[[T, int], int]
GetFun = typing.Callable[[T, C], X]
SetFun = typing.Callable[[T, C, X], None]

class ArrayPath(typing.Generic[X, T, C], Path[X]):

    def __init__(self, typ:int, dim:int, coord:C, parent: FamilyPath[T], size_fun:SizeCallable[T], get_fun:GetFun[T,C,X], set_fun:SetFun[T,C,X]) -> None:
        super().__init__(typ)
        self.coord = coord
        self.dim = dim
        self.parent = parent
        self.size_fun = size_fun
        self.get_fun = get_fun
        self.set_fun = set_fun

    def __getitem__(self, base: so.odbase) -> X:
        return self.get_in_array(self.parent[base])

    def get_in_array(self, array:T) -> X:
        return self.get_fun(array, self.coord)
    
    def __setitem__(self, base: so.odbase, value: X) -> None:
        self.set_fun(self.parent[base], self.coord, value)
    
    def check(self, base: so.odbase) -> bool:
        return self.parent.check(base) and self.check_index(self.parent[base])

    def check_index(self, array:T) -> bool:
        for _dim in range(self.dim):
            dim_size = self.size_fun(array, _dim)
            if self.coord[_dim] >= dim_size:
                return False
        return True

    def __str__(self) -> str:
        if self.dim == 1:
            return f"{self.parent}[{self.coord[0]}]"
        elif self.dim == 1:
            return f"{self.parent}[{self.coord[0]},{self.coord[1]}]"
        else:
            return f"{self.parent}[{self.coord[0]},{self.coord[1]},{self.coord[2]}]"

class GroupPath(typing.Generic[X, T], Path[X]):
    def __init__(self, typ:int, index:int, name:str, parent: Path[T], size_fun:typing.Callable[[T], int], get_fun:typing.Callable[[T, str], X], set_fun:typing.Callable[[T, str, X], None]) -> None:
        super().__init__(typ)
        self.index = index
        self.name = name
        self.parent = parent
        self.size_fun = size_fun
        self.get_fun = get_fun
        self.set_fun = set_fun

    def __getitem__(self, base: so.odbase) -> X:
        return self.get_in_group(self.parent[base])

    def __setitem__(self, base: so.odbase, value: X) -> None:
        self.set_fun(self.parent[base], self.name, value)

    def get_in_group(self, group:T) -> X:
        return self.get_fun(group, self.name)
    
    def __str__(self) -> str:
        return f"{self.parent}[{self.name}]"

    def check(self, base: so.odbase) -> bool:
        group = self.parent[base]
        return self.index < self.size_fun(group)

class BasePath(FamilyPath[so.odbase]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_base, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odbase:
        return so.odbase_get_odbase(base, self.name, self.pos)

    def __setitem__(self, base: so.odbase, value: so.odbase) -> None:
        so.odbase_put_odbase(self.parent[base], self.name, value, self.pos)

class FloatPath(FamilyPath[float]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_r0, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> float:
        return so.odbase_get_double(base, self.name, self.pos)

    def __setitem__(self, base: so.odbase, value: float) -> None:
        so.odbase_put_double(self.parent[base], self.name, value, self.pos)

class IntPath(FamilyPath[int]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_i0, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> int:
        return so.odbase_get_int(base, self.name, self.pos)
    
    def __setitem__(self, base: so.odbase, value: int) -> None:
        so.odbase_put_int(self.parent[base], self.name, value, self.pos)

class I1Path(FamilyPath[so.odi1]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_i1, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odi1:
        return so.odbase_get_odi1(base, self.name, self.pos)
    
    def __setitem__(self, base: so.odbase, value: so.odi1) -> None:
        for path in enumerate_i1(self, base):
            path[base] = path.get_in_array(value)

class I1ArrayPath(ArrayPath[int, so.odi1, C_1D]):
    def __init__(self, index: C_1D, parent: FamilyPath[so.odi1]) -> None:
        super().__init__(so.od_i0, 1, index, parent, (lambda array,dim: so.odi1_size(array)), lambda array, coord: so.odi1_get(array, coord[0]), lambda array, coord, value: so.odi1_put(array, coord[0], value))

class I2ArrayPath(ArrayPath[int, so.odi2, C_2D]):
    def __init__(self, index: C_2D, parent: FamilyPath[so.odi2]) -> None:
        super().__init__(so.od_i0, 2, index, parent, lambda array, dim : so.odi2_card(array, dim), lambda array, coord: so.odi2_get(array, coord[0], coord[1]), lambda array, coord, value:so.odi2_put(array, coord[0], coord[1],value))

class IgElementPath(GroupPath[int, so.odig]):
    def __init__(self, index:int, name: str, parent: Path[so.odig]) -> None:
        super().__init__(so.od_i0, index, name, parent, so.odig_size, so.odig_get, so.odig_put)

class I2Path(FamilyPath[so.odi2]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_i2, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odi2:
        return so.odbase_get_odi2(base, self.name, self.pos)
    
    def __setitem__(self, base: so.odbase, value: so.odi2) -> None:
        so.odbase_put_odi2(self.parent[base], self.name, value, self.pos)

class R1Path(FamilyPath[so.odr1]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_r1, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odr1:
        return so.odbase_get_odr1(base, self.name, self.pos)
    
    def __setitem__(self, base: so.odbase, value: so.odr1) -> None:
        for path in enumerate_r1(self, base):
            path[base] = path.get_in_array(value)

class RgElementPath(GroupPath[float, so.odrg]):
    def __init__(self, index:int,name: str, parent: Path[so.odrg]) -> None:
        super().__init__(so.od_r0, index, name, parent, so.odrg_size, so.odrg_get, so.odrg_put)

class R1ArrayPath(ArrayPath[float, so.odr1, C_1D]):
    def __init__(self, index: C_1D, parent: FamilyPath[so.odr1]) -> None:
        super().__init__(so.od_r0, 1, index, parent, r1_size, r1_get, r1_set)

class R2ArrayPath(ArrayPath[float, so.odr2, C_2D]):
    def __init__(self, index: C_2D, parent: FamilyPath[so.odr2]) -> None:
        super().__init__(so.od_r0, 2, index, parent, lambda array, dim : so.odr2_size(array, dim), lambda array, coord: so.odr2_get(array, coord[0], coord[1]), lambda array, coord, value:so.odr2_put(array, coord[0], coord[1],value))

class R3ArrayPath(ArrayPath[float, so.odr3, C_3D]):
    def __init__(self, index: C_3D, parent: FamilyPath[so.odr3]) -> None:
        super().__init__(so.od_r0, 3, index, parent, lambda array, dim : so.odr3_size(array, dim), lambda array, coord: so.odr3_get(array, coord[0], coord[1], coord[2]), lambda array, coord, value:so.odr3_put(array, coord[0], coord[1],coord[2],value))

class R2Path(FamilyPath[so.odr2]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_r2, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odr2:
        return so.odbase_get_odr2(base, self.name, self.pos)

    def __setitem__(self, base: so.odbase, value: so.odr2) -> None:
        for path in enumerate_r2(self, base):
            path[base] = path.get_in_array(value)

class R3Path(FamilyPath[so.odr3]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_r3, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odr3:
        return so.odbase_get_odr3(base, self.name, self.pos)

    def __setitem__(self, base: so.odbase, value: so.odr3) -> None:
        for path in enumerate_r3(self, base):
            path[base] = path.get_in_array(value)

class StrPath(FamilyPath[str]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_c0, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> str:
        return so.odbase_get_string(base, self.name, self.pos)

    def __setitem__(self, base: so.odbase, value: str) -> None:
        so.odbase_put_string(self.parent[base], self.name, value, self.pos)

class C1Path(FamilyPath[so.odc1]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_c1, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odc1:
        return so.odbase_get_odc1(base, self.name, self.pos)

    def __setitem__(self, base: so.odbase, value: so.odc1) -> None:
        for path in enumerate_c1(self, base):
            path[base] = path.get_in_array(value)

class C1ArrayPath(ArrayPath[str, so.odc1, C_1D]):
    def __init__(self, index: C_1D, parent: FamilyPath[so.odc1]) -> None:
        super().__init__(so.od_c0, 1, index, parent, lambda array, dim:so.odc1_size(array),lambda array, coord: so.odc1_get(array, coord[0]),lambda array, coord, value:so.odc1_put(array, coord[0], value))

class OdtPath(FamilyPath[so.odt]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_t, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odt:
        return so.odbase_get_odt(base, self.name, self.pos)

    def __setitem__(self, base: so.odbase, value: so.odt) -> None:
        so.odbase_put_odt(self.parent[base], self.name, value, self.pos)

class RgPath(FamilyPath[so.odrg]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_rg, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odrg:
        return so.odbase_get_odrg(base, self.name, self.pos)

    def __setitem__(self, base: so.odbase, value: so.odrg) -> None:
        so.odbase_put_odrg(self.parent[base], self.name, value, self.pos)
        
class IgPath(FamilyPath[so.odig]):

    def __init__(self, name: str, pos: int, parent: Path[so.odbase]) -> None:
        super().__init__(so.od_ig, name, pos, parent)

    def get_in_base(self, base: so.odbase) -> so.odig:
        return so.odbase_get_odig(base, self.name, self.pos)

    def __setitem__(self, base: so.odbase, value: so.odig) -> None:
        so.odbase_put_odig(self.parent[base], self.name, value, self.pos)

def r1_size(array, dim):
    return so.odr1_size(array)

def r1_get(array, coord):
    return so.odr1_get(array, coord[0])

def r1_set(array, coord, value):
    return so.odr1_put(array, coord[0], value)

def enumerate_i1(path:Path[so.odi1], base:so.odbase) -> typing.Iterable[I1ArrayPath]:
    yield from _enumerate_array(path, base, lambda array, dim:so.odi1_size(array), I1ArrayPath, 1)

def enumerate_i2(path:Path[so.odi2], base:so.odbase) -> typing.Iterable[I2ArrayPath]:
    yield from _enumerate_array(path, base, lambda array, dim:so.odi2_card(array, dim), I2ArrayPath, 2)

def enumerate_r1(path:Path[so.odr1], base:so.odbase) -> typing.Iterable[R1ArrayPath]:
    yield from _enumerate_array(path, base, lambda array, dim:so.odr1_size(array), R1ArrayPath, 1)

def enumerate_r2(path:Path[so.odr2], base:so.odbase) -> typing.Iterable[R2ArrayPath]:
    yield from _enumerate_array(path, base, lambda array, dim:so.odr2_size(array, dim), R2ArrayPath, 2)

def enumerate_r3(path:Path[so.odr3], base:so.odbase) -> typing.Iterable[R3ArrayPath]:
    yield from _enumerate_array(path, base, lambda array, dim:so.odr3_size(array, dim), R3ArrayPath, 3)

def enumerate_c1(path:Path[so.odc1], base:so.odbase) -> typing.Iterable[C1ArrayPath]:
    yield from _enumerate_array(path, base, lambda array, dim:so.odc1_size(array), C1ArrayPath, 1)

def enumerate_rg(path:Path[so.odrg], base:so.odbase) -> typing.Iterable[RgElementPath]:
    yield from _enumerate_group(path, base, so.odrg_card, so.odrg_name, RgElementPath)

def enumerate_ig(path:Path[so.odig], base:so.odbase) -> typing.Iterable[IgElementPath]:
    yield from _enumerate_group(path, base, so.odig_card, so.odig_name, IgElementPath)

def _enumerate_group(path, base, size_fun, name_fun, clazz) -> typing.Iterable:
    group = path[base]
    count = size_fun(group)
    for index in range(count):
        name = name_fun(group, index)
        yield clazz(index, name, path)

def _enumerate_array(path, base, size_fun, clazz, dims) -> typing.Iterable:
    array = path[base]
    sizes = []
    for dim in range(dims):
        sizes.append(size_fun(array, dim))
    
    for dim in range(dims):
        for coords in itertools.product(*[range(size) for size in sizes]):
            yield clazz(coords, path)

def enumerate_base_family_rec(path:Path[so.odbase], base:so.odbase) -> typing.Iterable[FamilyPath]:
    for new_path in enumerate_base(path, base):
        yield new_path
        typ = new_path.typ
        if typ == so.od_base:
            yield from enumerate_base_family_rec(new_path, base)

def enumerate_base_rec(path:Path[so.odbase], base:so.odbase) -> typing.Iterable[Path]:
    for new_path in enumerate_base(path, base):
        yield new_path
        typ = new_path.typ
        if typ == so.od_base:
            yield from enumerate_base_rec(new_path, base)
        elif typ == so.od_c1:
            yield from enumerate_c1(new_path, base)
        elif typ == so.od_r1:
            yield from enumerate_r1(new_path, base)
        elif typ == so.od_r2:
            yield from enumerate_r2(new_path, base)
        elif typ == so.od_r3:
            yield from enumerate_r3(new_path, base)
        elif typ == so.od_i1:
            yield from enumerate_i1(new_path, base)
        elif typ == so.od_i2:
            yield from enumerate_i2(new_path, base)
        elif typ == so.od_ig:
            yield from enumerate_ig(new_path, base)
        elif typ == so.od_rg:
            yield from enumerate_rg(new_path, base)

def enumerate_literals_rec(path:Path[so.odbase],base:so.odbase) -> typing.Iterable[Path]:
    for new_path in enumerate_base(path, base):
        typ = new_path.typ
        if typ == so.od_base:
            yield from enumerate_literals_rec(new_path, base)
        elif typ == so.od_c0:
            yield new_path
        elif typ == so.od_r0:
            yield new_path
        elif typ == so.od_i0:
            yield new_path
        elif typ == so.od_t:
            yield new_path
        elif typ == so.od_c1:
            yield from enumerate_c1(new_path, base)
        elif typ == so.od_r1:
            yield from enumerate_r1(new_path, base)
        elif typ == so.od_r2:
            yield from enumerate_r2(new_path, base)
        elif typ == so.od_r3:
            yield from enumerate_r3(new_path, base)
        elif typ == so.od_i1:
            yield from enumerate_i1(new_path, base)
        elif typ == so.od_i2:
            yield from enumerate_i2(new_path, base)
        elif typ == so.od_ig:
            yield from enumerate_ig(new_path, base)
        elif typ == so.od_rg:
            yield from enumerate_rg(new_path, base)
        else:
            raise Exception(f"Unhandled typ at path {new_path} : {flty.typ_names[typ]}")


def enumerate_base(path:BasePath[so.odbase],base:so.odbase) -> typing.Iterable[FamilyPath]:
        local_base = path[base]
        family_count = so.odbase_family_number(local_base)
        for family_index in range(1 - so.odessa_shift(), family_count + 1 - so.odessa_shift()):
            family_name = so.odbase_name(local_base, family_index).strip()
            typ = so.odbase_type(local_base, family_name)
            for index in range(1 - so.odessa_shift(), so.odbase_size(local_base, family_name) + 1 - so.odessa_shift()):
                if typ == so.od_base:
                    yield BasePath(family_name, index, path)
                elif typ == so.od_i0:
                    yield IntPath(family_name, index, path)
                elif typ == so.od_c0:
                    yield StrPath(family_name, index, path)
                elif typ == so.od_c1:
                    yield C1Path(family_name, index, path)
                elif typ == so.od_t:
                    yield OdtPath(family_name, index, path)
                elif typ == so.od_r0:
                    yield FloatPath(family_name, index, path)
                elif typ == so.od_r1:
                    yield R1Path(family_name, index, path)
                elif typ == so.od_r2:
                    yield R2Path(family_name, index, path)
                elif typ == so.od_r3:
                    yield R3Path(family_name, index, path)
                elif typ == so.od_i1:
                    yield I1Path(family_name, index, path)
                elif typ == so.od_i2:
                    yield I2Path(family_name, index, path)
                elif typ == so.od_i2:
                    yield I2Path(family_name, index, path)
                elif typ == so.od_rg:
                    yield RgPath(family_name, index, path)
                elif typ == so.od_ig:
                    yield IgPath(family_name, index, path)
                else:
                    raise NotImplementedError(f"Unhandled typ : {flty.typ_names[typ]}")
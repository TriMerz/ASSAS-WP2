import pyastec as pyas
from typing import Iterable
import itertools
import fluent.cache as cache
import fluent.path as pa
import fluent.family as fl
import fluent.fluent_odessa as flod

def enumerate_base_family(family:fl.BaseFamily, base:cache.CachedOptOdbase)-> Iterable[pa.BaseFamilyPath]:
    yield from enumerate_base_family_inline(family, family.parent.get_from(base))

def enumerate_base_family_inline(family:fl.BaseFamily, base:pyas.odbase)-> Iterable[pa.BaseFamilyPath]:
    yield from enumerate_family(family, base) #type: ignore

def enumerate_family(family:fl.Family[flod.T], base:cache.CachedOptOdbase) -> Iterable[pa.FamilyPath[flod.T]]:
    inline_base = family.parent.get_from(base)
    yield from enumerate_family_inline(family, inline_base)

def enumerate_family_inline(family:fl.Family[flod.T], inline_base:pyas.odbase) -> Iterable[pa.FamilyPath[flod.T]]:
    family_size = pyas.odbase_size(inline_base, family.name)

    for index in range(1 - pyas.odessa_shift(), family_size + 1 - pyas.odessa_shift()):
        yield family.path(index)

def enumerate_families(path:pa.BasePath, base:cache.CachedOptOdbase) -> Iterable[fl.Family]:
    inline_base = path.get_from(base)
    yield from enumerate_families_inline(path, inline_base)
        
def enumerate_families_inline(path:pa.BasePath, inline_base:pyas.odbase) -> Iterable[fl.Family]:
    size = pyas.odbase_family_number(inline_base)

    family_builder = {
        pyas.od_base: fl.BaseFamily,
        pyas.od_c0: fl.StrFamily,
        pyas.od_c1: fl.C1Family,
        pyas.od_i0: fl.IntFamily,
        pyas.od_i1: fl.I1Family,
        pyas.od_i2: fl.I2Family,
        pyas.od_ig: fl.IGFamily,
        pyas.od_r0: fl.FloatFamily,
        pyas.od_r1: fl.R1Family,
        pyas.od_r2: fl.R2Family,
        pyas.od_r3: fl.R3Family,
        pyas.od_rg: fl.RGFamily,
        pyas.od_t: fl.TextFamily
    }

    for pos in range(1 - pyas.odessa_shift(), size + 1 - pyas.odessa_shift()):
        name = pyas.odbase_name(inline_base, pos).strip()
        typ = pyas.odbase_type(inline_base, name)
        if typ in family_builder:
            yield family_builder[typ](name, path)
        else:
            raise Exception(f"Unhandled type : {flod.typ_names[typ]}")

def enumerate_c1(path: pa.C1FamilyPath, base:pyas.odbase) -> Iterable[pa.C1ElementPath]:
    yield from _enumerate_array(path, base, odc1_size_wrapper, pa.C1ElementPath, 1)

def enumerate_i1(path: pa.I1FamilyPath, base:pyas.odbase) -> Iterable[pa.I1ElementPath]:
    yield from _enumerate_array(path, base, odi1_size_wrapper, pa.I1ElementPath, 1)

def enumerate_i2(path: pa.I2FamilyPath, base:pyas.odbase) -> Iterable[pa.I2ElementPath]:
    yield from _enumerate_array(path, base, pyas.odi2_card, pa.I2ElementPath, 2)

def enumerate_r1(path: pa.R1FamilyPath, base:pyas.odbase) -> Iterable[pa.R1ElementPath]:
    yield from _enumerate_array(path, base, odr1_size_wrapper, pa.R1ElementPath, 1)

def enumerate_r2(path: pa.R2FamilyPath, base:pyas.odbase) -> Iterable[pa.R2ElementPath]:
    yield from _enumerate_array(path, base, pyas.odr2_size, pa.R2ElementPath, 2)

def enumerate_r3(path: pa.R3FamilyPath, base:pyas.odbase) -> Iterable[pa.R3ElementPath]:
    yield from _enumerate_array(path, base, pyas.odr3_size, pa.R3ElementPath, 3)

def _enumerate_array(path:pa.ArrayPath, base:pyas.odbase, size_fun, builder, dims:int):
    array = path.get_from(base)
    return _enumerate_array_inline(path, array, size_fun, builder, dims)

def _enumerate_array_inline(path:pa.ArrayPath, array:flod.AT, size_fun, builder, dims:int):
    ranges = []
    for dim in range(dims):
        ranges.append(range(size_fun(array, dim)))
    for coords in itertools.product(*ranges):
        yield builder(coords, path)

def enumerate_ig(path:pa.IGFamilyPath, base:pyas.odbase) -> Iterable[pa.IGElementPath]:
    yield from _enumerate_group(path, base, pyas.odig_size, pyas.odig_name, pa.IGElementPath)

def enumerate_rg(path:pa.RGFamilyPath, base:pyas.odbase) -> Iterable[pa.RGElementPath]:
    yield from _enumerate_group(path, base, pyas.odrg_size, pyas.odrg_name, pa.RGElementPath)

def _enumerate_group(path:pa.GroupPath, base:pyas.odbase, size_fun, name_fun, builder):
    group = path.get_from(base)
    return _enumerate_group_inline(path, group, size_fun, name_fun, builder)

def _enumerate_group_inline(path:pa.GroupPath, group:flod.GT, size_fun, name_fun, builder):
    size = size_fun(group)

    for pos in range(1 - pyas.odessa_shift(), size + 1 - pyas.odessa_shift(),):
        name = name_fun(group, pos)
        yield builder(name, path)

def enumerate_base(path:pa.BasePath, base:pyas.odbase) -> Iterable[pa.FamilyPath]:
    inline_base = path.get_from(base)
    yield from enumerate_base_inline(path, inline_base)

def enumerate_base_inline(path:pa.BasePath, base:pyas.odbase) -> Iterable[pa.FamilyPath]:
    for family in enumerate_families_inline(path, base):
        yield from enumerate_family_inline(family, base) 


def enumerate_paths_rec(path:pa.BasePath, base:cache.CachedOptOdbase) -> Iterable[pa.FamilyPath]:
    inline_base = path.get_from(base)
    yield from enumerate_paths_rec_inline(path, inline_base)


def enumerate_paths_rec_inline(path:pa.BasePath, base:pyas.odbase) -> Iterable[pa.FamilyPath]:
    for family_path in enumerate_base_inline(path, base):
        yield family_path
        if isinstance(family_path, pa.BaseFamilyPath):
            inline_base = family_path.get_from_inline(base)
            yield from enumerate_paths_rec_inline(family_path, inline_base)



def enumerate_literals_rec(path:pa.BasePath, base:pyas.odbase) -> Iterable[pa.Path]:

    literals = [pyas.od_c0, pyas.od_r0, pyas.od_i0]

    literal_containers = {
        pyas.od_c1:enumerate_c1,
        pyas.od_i1:enumerate_i1,
        pyas.od_i2:enumerate_i2,
        pyas.od_r1:enumerate_r1,
        pyas.od_r2:enumerate_r2,
        pyas.od_r3:enumerate_r3,
        pyas.od_ig:enumerate_ig,
        pyas.od_rg:enumerate_rg,
    }

    inline_base = path.get_from(base)

    for family_path in enumerate_base_inline(path, inline_base):
        typ = family_path.typ
        if typ in literals:
            yield family_path
        elif typ in literal_containers:
            yield from literal_containers[typ](family_path, base)

def odr1_size_wrapper(array:pyas.odr1, dim:int):
    return pyas.odr1_size(array)

def odc1_size_wrapper(array:pyas.odc1, dim:int):
    return pyas.odc1_size(array)

def odi1_size_wrapper(array:pyas.odi1, dim:int):
    return pyas.odi1_size(array)
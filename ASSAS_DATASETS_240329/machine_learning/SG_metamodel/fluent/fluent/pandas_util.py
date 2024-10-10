from typing import Iterable, Tuple, Callable
import pyastec as pyas
import logging
import pandas as pd
import fluent.path as pa
import fluent.enumerate as en
import fluent.cache as cache
import fluent.fluent_odessa as flod
import numpy as np

IGNORED_FAMILIES = ["CREE"]
DUOS = ["T_liq", "T_gas", "x_alfa", "P_steam", "P_o2", "P_h2", "P_n2", "P_bho2", "P_co2", "v_liq", "v_gas", "FLUX" , "P" , "T_wall" , "x_alfa1" , "WTEMP", "VG", "VF", "TLIQ", "TFLU", "MASS", "PRES", "ZVELO", "VLIQ", "FPHDDRY", "FPHDWET", "FPHGAS", "FPHAERO", "FPHWATER", "ZHEWALL", "ZHEZONE", "ZHEZONS"]
TIME_FIRST = ["HTEM", "FLOW"]
FULL_ARRAYS = ["C", "SNGZ", "TEA", "TRTSURF", "VIEW", "DEA"]
IGNORE_FLOAT_ARRAY = ["P_vol"]
INDEXES = ["NAME", "SMAT"]
DISCRETE = ["STAT", "ZPRTNA", "ZPHASE", "UNIT", "HOLS" , "CTYPF"]
ACCEPTABLE_SUB_TYPES = [pyas.od_i0, pyas.od_r0, pyas.od_rg, pyas.od_base, pyas.od_c0, pyas.od_r1]

def pandas_enumerate_str(path:pa.StrFamilyPath,
                         discrete_names:list[str] = DISCRETE,
                         indexes_names:list[str] = INDEXES
                         )->Iterable[pa.StrFamilyPath]:
    if path.name in discrete_names:
        return [path]
    elif not path.name in indexes_names:
        logging.debug(f"Unrecognized str at path {path}, ignoring it")
    return []
    
def pandas_enumerate_r1(path:pa.R1FamilyPath,
                         base:cache.CachedOptOdbase,
                         duos:list[str] = DUOS,
                         time_first_arrays:list[str] = TIME_FIRST,
                         full_arrays:list[str] = FULL_ARRAYS,
                         ignored_r1:list[str] = IGNORE_FLOAT_ARRAY) -> Iterable[pa.R1ElementPath]:
    array = path.get_from(base)
    size = pyas.odr1_size(array)
    if path.name in duos:
        if size == 2:
            yield pa.R1ElementPath((2 - pyas.odessa_shift(),), path)
        else:
            raise Exception(f"DUO float array {path} has size {size} instead of 2")
    elif path.name in time_first_arrays:
        for index in range(2 - pyas.odessa_shift(), size + 1 - pyas.odessa_shift()):
            yield pa.R1ElementPath((index,), path)
    elif path.name in full_arrays:
        for index in range(1 - pyas.odessa_shift(), size + 1 - pyas.odessa_shift()):
            yield pa.R1ElementPath((index,), path)
    elif not path.name in ignored_r1:
        raise Exception(f"path {path} is not DUO, not TIME_FIRST, not FULL nor IGNORED, handle it !")

def pandas_enumerate_base(base:cache.CachedOptOdbase,
                           root:pa.BasePath = pa.Root(),
                           acceptable_sub_types:list[int] = ACCEPTABLE_SUB_TYPES) -> Iterable[pa.Path]:
    for path in en.enumerate_paths_rec(root, base):
        if path.typ not in acceptable_sub_types:
            logging.debug(f"Ignoring path {path} : not in acceptable types")
        if isinstance(path, pa.FloatFamilyPath):
            yield path
        elif isinstance(path, pa.StrFamilyPath):
            yield from pandas_enumerate_str(path)
        elif isinstance(path, pa.R1FamilyPath):
            yield from pandas_enumerate_r1(path, base)

def to_df(bases: Iterable[Tuple[str, float, Iterable[Tuple[pa.Path, flod.T]]]], count:int, time_key:str="t"):
    dict_base = {time_key:np.full(count, np.nan, dtype=float)}

    index = 0

    for (dir, time, paths) in bases:
        for (path, value) in paths:
            path_str = path.odessa_repr
            
            logging.debug(f"Reading {path_str}, found {value}")

            if path_str not in dict_base:
                if isinstance(value, str):
                    filler = np.full(count, None, dtype=str)
                elif isinstance(value, int):
                    filler = np.full(count, np.nan, dtype=int)
                else:
                    filler = np.full(count, np.nan, dtype=float)
                dict_base[path_str] = filler

            dict_base[path_str][index] = value
        
        dict_base[time_key][index] = time

        index += 1
    
    df = pd.DataFrame.from_dict(dict_base)

    df = df.set_index(time_key)

    return df

def convert_object_to_catagorical(df: pd.DataFrame):
    object_columns = df.columns[df.dtypes == object]
    df[object_columns] = df[object_columns].astype('category')


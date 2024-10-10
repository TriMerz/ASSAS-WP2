import pyastec as pyas
import pandas as pd
import fluent.path as flpa
from  fluent.astec import ROOT
import typing
import logging

TIME_INDEX_KEY = "time"

DEFAULT_DUOS = [ "T_liq", "T_gas", "x_alfa", "P_steam", "P_o2", "P_h2", "P_n2", "P_bho2", "P_co2", "v_liq", "v_gas", "FLUX" , "P" , "T_wall" , "x_alfa1" , "WTEMP", "VG", "VF", "TLIQ", "TFLU", "MASS", "PRES", "ZVELO", "VLIQ", "FPHDDRY", "FPHDWET", "FPHGAS", "FPHAERO", "FPHWATER"]

def filter_duos(path:flpa.Path, duos = DEFAULT_DUOS):
    if isinstance(path, flpa.R1ArrayPath):
        # Array of floats
        if any([duo == path.parent.name for duo in duos]):
            # Name is on of the recorded duo path
            return not path.coord[0] == 0
    return True

def filter_names(path:flpa.Path):
    if isinstance(path, flpa.StrPath):
        return path.name != "NAME"
    return True

def to_df(dir:str, times:list[float], filter:typing.Callable[[flpa.Path], bool]=lambda x:True)->pd.DataFrame:

    values = {TIME_INDEX_KEY:[]}

    # Number of values to fill if key not present

    for (index, time) in enumerate(times):
        logging.info(f"Converting time {time}")
        base = pyas.odloaddir(dir, time)

        written = {name:False for name, value in values.items()}
        
        values[TIME_INDEX_KEY].append(time)
        written[TIME_INDEX_KEY] = True
        
        for path in flpa.enumerate_literals_rec(ROOT, base):
            if(filter(path)):
                name = str(path)
                value = path[base]
                if name in values:
                    values[name].append(value)
                    written[name] = True
                else:
                    if isinstance(value, str):
                        dummy = ''
                    elif isinstance(value, float):
                        dummy = 0.
                    else:
                        dummy = 0
                    list = [dummy] * index
                    list.append(value)
                    values[name] = list
                    written[name] = True
        
        for name, written in written.items():
            if not written:
                value = values[name]
                if isinstance(value, str):
                    value.append("")
                elif isinstance(value, float):
                    value.append(0.)
                else:
                    value.append(0)

    df = pd.DataFrame.from_dict(values)
    df = df.set_index(TIME_INDEX_KEY)

    return df
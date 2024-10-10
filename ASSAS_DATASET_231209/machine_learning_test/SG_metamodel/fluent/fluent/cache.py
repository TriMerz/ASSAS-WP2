import pyastec as pyas
import typing

class CachedOdbase():

    def __init__(self, base:pyas.odbase) -> None:
        self.base = base
        self.cache = {}

CachedOptOdbase = typing.Union[pyas.odbase, CachedOdbase]

def cached(base:pyas.odbase):
    return CachedOdbase(base)
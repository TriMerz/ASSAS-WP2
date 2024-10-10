import full_pyastec as full_pyas
import typing

class CachedOdbase():

    def __init__(self, base:full_pyas.odbase) -> None:
        self.base = base
        self.cache = {}

CachedOptOdbase = typing.Union[full_pyas.odbase, CachedOdbase]

def cached(base:full_pyas.odbase):
    return CachedOdbase(base)

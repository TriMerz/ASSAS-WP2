import swigodessa as so
import fluent.path as flpa

class Filter:

    def __init__(self, base:so.odbase) -> None:
        self.base = base

    def filter(self, path:flpa.Path) -> bool:
        return False

import fluent.path as pa
import fluent.fluent_astec as flas

class Connecti(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: flas.AstecRoot=flas.AstecRoot()) -> None:
        super().__init__("CONNECTI", pos, parent)

    def stat(self):
        return Stat(self)
    
    def name_(self):
        return Name(self)
    
    def from_(self):
        return From(self)
    
    def to(self):
        return To(self)
    
    def h(self, pos:int):
        return H(pos, self)

class Stat(pa.StrFamilyPath):
    def __init__(self, parent:Connecti) -> None:
        super().__init__("STAT", 0, parent)

class Name(pa.StrFamilyPath):
    def __init__(self, parent:Connecti) -> None:
        super().__init__("NAME", 0, parent)

class From(pa.StrFamilyPath):
    def __init__(self, parent:Connecti) -> None:
        super().__init__("FROM", 0, parent)

class To(pa.StrFamilyPath):
    def __init__(self, parent:Connecti) -> None:
        super().__init__("TO", 0, parent)

class H(pa.TextFamilyPath):
    def __init__(self, pos:int, parent:Connecti) -> None:
        super().__init__("H", pos, parent)

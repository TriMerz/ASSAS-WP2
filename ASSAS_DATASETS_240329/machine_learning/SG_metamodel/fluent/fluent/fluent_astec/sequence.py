import fluent.path as pa
import fluent.fluent_astec as flas
import pyastec as pyas

class Sequence(pa.BaseFamilyPath):
    def __init__(self, parent:flas.AstecRoot = flas.AstecRoot()) -> None:
        super().__init__("SEQUENCE", 1 - pyas.odessa_shift(), parent)

    def tscram(self):
        return Tscram(self)
    
    def step(self):
        return Step(self)   

    def time(self):
        return Time(self)
    
    def tima(self):
        return Tima(self)
    
    def iter0(self):
        return Iter0(self)
    
    def iter(self):
        return Iter(self)
    
    def lookunit(self):
        return Lookunit(self)
    
    def cputime(self):
        return Cputime(self)

class Tscram(pa.FloatFamilyPath):
    def __init__(self, parent: Sequence = Sequence()) -> None:
        super().__init__("TSCRAM", 1 - pyas.odessa_shift(), parent)

class Step(pa.FloatFamilyPath):
    def __init__(self, parent: Sequence = Sequence()) -> None:
        super().__init__("STEP", 1 - pyas.odessa_shift(), parent)

class Time(pa.FloatFamilyPath):
    def __init__(self, parent: Sequence = Sequence()) -> None:
        super().__init__("TIME", 1 - pyas.odessa_shift(), parent)

class Tima(pa.FloatFamilyPath):
    def __init__(self, parent: Sequence = Sequence()) -> None:
        super().__init__("TIMA", 1 - pyas.odessa_shift(), parent)

class Iter0(pa.IntFamilyPath):
    def __init__(self, parent: Sequence = Sequence()) -> None:
        super().__init__("ITER0", 1 - pyas.odessa_shift(), parent)

class Iter(pa.IntFamilyPath):
    def __init__(self, parent: Sequence = Sequence()) -> None:
        super().__init__("ITER", 1 - pyas.odessa_shift(), parent)

class Lookunit(pa.IntFamilyPath):
    def __init__(self, parent: Sequence = Sequence()) -> None:
        super().__init__("LOOKUNIT", 1 - pyas.odessa_shift(), parent)

class Cputime(pa.FloatFamilyPath):
    def __init__(self, parent: Sequence = Sequence()) -> None:
        super().__init__("CPUTIME", 1 - pyas.odessa_shift(), parent)


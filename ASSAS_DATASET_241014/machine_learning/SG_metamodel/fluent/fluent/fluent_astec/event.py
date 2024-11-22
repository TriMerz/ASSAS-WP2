import fluent.path as pa
import fluent.fluent_astec as flas

class Event(pa.BaseFamilyPath):
    def __init__(self, pos: int, parent: flas.AstecRoot=flas.AstecRoot()) -> None:
        super().__init__("EVENT", pos, parent)

import fluent.path as flpa

def flatten(path:flpa.Path):
    if isinstance(path, (flpa.FamilyPath, flpa.ArrayPath, flpa.GroupPath)):
        yield from flatten(path.parent)
    yield path

def descends_from(path:flpa.Path, _from:flpa.Path)->bool:
    for _from_flattened, path_flattened in zip(flatten(_from), flatten(path)):
        if _from_flattened != path_flattened:
            return False
    return True
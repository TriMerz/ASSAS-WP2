import fluent.path as flpa
import pyastec as pyas
import logging

MACR_STATUT = ['COMPACT', 'PERFORAT', 'CRACKED','DISLOCAT', 'ABSENT']

def fill(root:flpa.Path, filled:pyas.odbase, filler:pyas.odbase):
    """
    Fills all values of the base filler inside the filled base from path root.
    Root is relative to both filled and filler (namely filled and filler should be at the same depth)
    """
    for path in flpa.enumerate_base(root, filler):
        parent = path.parent[filled]

        filled_family_size = pyas.odbase_size(parent, path.name)
        filler_family_size = pyas.odbase_size(path.parent[filler], path.name)
        for index in range(1-pyas.odessa_shift(), filled_family_size - filler_family_size + 1 - pyas.odessa_shift()):
            logging.debug(f"filled family size {filled_family_size} > filler family size {filler_family_size}, deleting {path} at index {filled_family_size - index - 1}")
            pyas.odbase_delete_element(parent, path.name, filled_family_size - 1 - index)

        if isinstance(path, flpa.BasePath):
            if path.name == "MACR":
                for statut in MACR_STATUT:
                    if pyas.odbase_size(path[filled], statut) > 0:
                        logging.debug(f"Deleting statut {statut} of macr {path}")
                        pyas.odbase_delete_element(path[filled], statut, 1 - pyas.odessa_shift())
        
            if path.name == "MESH":
                if pyas.odbase_size(path[filled], "COMPRADR") > 0:
                    logging.debug(f"Deleting COMPRADR for path {path}")
                    pyas.odbase_delete_family(path[filled], "COMPRADR")
            if path.pos >= filled_family_size + 1 - pyas.odessa_shift():
                pyas.odbase_put_odbase(parent, path.name, pyas.odbase_init(), path.pos)
            fill(path, filled, filler)
        elif isinstance(path, flpa.FloatPath):
            pyas.odbase_put_double(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.IntPath):
            pyas.odbase_put_int(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.StrPath):
            pyas.odbase_put_string(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.R1Path):
            pyas.odbase_put_odr1(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.R2Path):
            pyas.odbase_put_odr2(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.R3Path):
            pyas.odbase_put_odr3(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.I1Path):
            pyas.odbase_put_odi1(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.I2Path):
            pyas.odbase_put_odi2(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.OdtPath):
            pyas.odbase_put_odt(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.RgPath):
            pyas.odbase_put_odrg(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.IgPath):
            pyas.odbase_put_odig(parent, path.name, path[filler], path.pos)
        elif isinstance(path, flpa.C1Path):
            pyas.odbase_put_odc1(parent, path.name, path[filler], path.pos)
        else:
            raise NotImplementedError(f"Path {path} not handled")

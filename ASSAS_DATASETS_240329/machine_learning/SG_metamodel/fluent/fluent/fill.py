
import pyastec as pyas
import logging
import fluent.path as pa
import fluent.enumerate as en

MACR_STATUT = ['COMPACT', 'PERFORAT', 'CRACKED','DISLOCAT', 'ABSENT']

def fill(root:pa.BasePath, filled:pyas.odbase, filler:pyas.odbase):
    """
    Fills all values of the base filler inside the filled base from path root.
    Root is relative to both filled and filler (namely filled and filler should be at the same depth)
    """
    for path in en.enumerate_base(root, filler):
        filled_parent = root.get_from(filled)
        filler_parent = root.get_from(filler)

        filled_family_size = pyas.odbase_size(filled_parent, path.name)
        filler_family_size = pyas.odbase_size(filler_parent, path.name)
        for index in range(1 - pyas.odessa_shift(), filled_family_size - filler_family_size + 1 - pyas.odessa_shift()):
            logging.debug(f"filled family size {filled_family_size} > filler family size {filler_family_size}, deleting {path} at index {filled_family_size - index - 1}")
            pyas.odbase_delete_element(filled_parent, path.name, filled_family_size - 1 - index)

        if isinstance(path, pa.BaseFamilyPath):
            if path.name == "MACR":
                for statut in MACR_STATUT:
                    macr = path.get_from(filled)
                    if pyas.odbase_size(macr, statut) > 0:
                        logging.debug(f"Deleting statut {statut} of macr {path}")
                        pyas.odbase_delete_element(macr, statut, 1 - pyas.odessa_shift())
        
            if path.name == "MESH":
                mesh = path.get_from(filled)
                if pyas.odbase_size(mesh, "COMPRADR") > 0:
                    logging.debug(f"Deleting COMPRADR for path {path}")
                    pyas.odbase_delete_family(mesh, "COMPRADR")
            if path.pos >= filled_family_size:
                pyas.odbase_put_odbase(filled_parent, path.name, pyas.odbase_init(), path.pos)
            fill(path, filled, filler)
        else:
            path.put_from(filled, path.get_from(filler))

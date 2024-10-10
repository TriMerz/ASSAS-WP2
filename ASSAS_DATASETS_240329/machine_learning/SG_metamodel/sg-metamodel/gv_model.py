from typing import Iterable
import pickle
import os
import pyastec as pyas

METADATA_FILE = "metadata.pkl"
MAPPING_FILE = "mapping.pkl"

PRIMARY_ROOT = "PRIMARY"
SECONDAR_ROOT = "SECONDAR"
CONNECTI_ROOT = "CONNECTI"
SENSOR_ROOT = "SENSOR"

GVS = [1, 2, 3, 4]
HOT_BOX_PREFIX = "HB"

class NameMapping:
    def __init__(self,
                 primary_volume_names: dict[int, str],
                 secondar_volume_names: dict[int, str],
                 primary_junction_names: dict[int, str],
                 secondar_junction_names: dict[int, str],
                 primary_wall_names: dict[int, str],
                 secondar_wall_names: dict[int, str],
                 connecti_names: dict[int, str],
                 sensor_names: dict[int, str]) -> None:
        self.primary_volume_names = primary_volume_names
        self.secondar_volume_names = secondar_volume_names
        self.primary_junction_names = primary_junction_names
        self.secondar_junction_names = secondar_junction_names
        self.primary_wall_names = primary_wall_names
        self.secondar_wall_names = secondar_wall_names
        self.connecti_names = connecti_names
        self.sensor_names = sensor_names

        self.primary_volume_indexes = self.inverse(self.primary_volume_names)
        self.secondar_volume_indexes = self.inverse(self.secondar_volume_names)
        self.primary_junction_indexes = self.inverse(
            self.primary_junction_names)
        self.secondar_junction_indexes = self.inverse(
            self.secondar_junction_names)
        self.primary_wall_indexes = self.inverse(self.primary_wall_names)
        self.secondar_wall_indexes = self.inverse(self.secondar_wall_names)
        self.connecti_indexes = self.inverse(self.connecti_names)
        self.sensor_indexes = self.inverse(self.sensor_names)

    def inverse(self, dict):
        return {v: k for k, v in dict.items()}

    def __str__(self) -> str:
        return f"Primary volume : {self.primary_volume_names}\n \
        Primary junction : {self.primary_junction_names}\n \
        Primary volume : {self.primary_volume_names}\n \
        Primary wall : {self.primary_wall_names}\n \
        Secondar junction : {self.secondar_junction_names}\n \
        Secondar volume : {self.secondar_wall_names}\n \
        Secondar wall : {self.secondar_wall_names}\n \
        Connectis : {self.connecti_names}\n \
        Sensors : {self.sensor_names}\n"


class ObjectMetadata:
    def __init__(self, root: str, name: str, common_name: str, gv: int) -> None:
        self.root = root
        self.name = name
        self.common_name = common_name
        self.gv = gv


class PathMetadata:
    def __init__(self, name: str, object: ObjectMetadata, input: bool, output: bool) -> None:
        self.name = name
        self.object = object
        self.input = input
        self.output = output
        self.gv_name = f"{self.object.root}:{self.object.common_name}:{self.name}"

    def __str__(self) -> str:
        return f"{self.name} - {'I' if self.input else ''}{'O' if self.output else ''}"

    def __repr__(self) -> str:
        return self.__str__()

    def get_gv_name(self):
        return self.gv_name


class GvModelMetadata:

    def __init__(self,
                 name_mapping:NameMapping,
                 metadatas: list[PathMetadata],
                 input_index_to_name: dict[int, str],
                 input_name_to_index: dict[str, int],
                 output_index_to_name: dict[int, str],
                 output_name_to_index: dict[str, int],
                 input_indexes: list[int],
                 output_indexes: list[int],) -> None:
        self.name_mapping = name_mapping
        self.metadata = metadatas
        self.input_index_to_name = input_index_to_name
        self.input_name_to_index = input_name_to_index
        self.output_index_to_name = output_index_to_name
        self.output_name_to_index = output_name_to_index
        self.input_indexes = input_indexes
        self.output_indexes = output_indexes
        self.output_index_to_input_index = {}
        self.input_index_to_output_index = {}

        for input_index in input_indexes:
            name = self.input_index_to_name[input_index]
            if name in self.output_name_to_index:
                output_index = self.output_name_to_index[name]
                self.output_index_to_input_index[output_index] = input_index
            else:
                output_index = None
            self.input_index_to_output_index[input_index] = output_index

def generate_sensor_datas(gv: int) -> Iterable[ObjectMetadata]:
    yield ObjectMetadata(SENSOR_ROOT, f"TX_GV{gv}", "TX_GVX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"NGE_GV{gv}", "NGE_GVX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"NGL_GV{gv}", "NGL_GVX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"ML_GV{gv}", "ML_GVX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"MV_GV{gv}", "MV_GVX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"PUIS_GV{gv}", "PUIS_GVX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"QEGV{gv}", "QEGVX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"QSGV{gv}", "QSGVX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"P_GV{gv}", "P_GVX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"PGV_TBF{gv}", "PGV_TBFX", gv)
    yield ObjectMetadata(SENSOR_ROOT, f"PGV_STA{gv}", "PGV_STAX", gv)


def generate_connecti_datas(gv: int) -> Iterable[ObjectMetadata]:
    yield ObjectMetadata(CONNECTI_ROOT, f"RE_GV{gv}", "RE_GVX", gv)
    yield ObjectMetadata(CONNECTI_ROOT, f"ASGGV{gv}", "ASGGVX", gv)
    yield ObjectMetadata(CONNECTI_ROOT, f"AREGV{gv}", "AREGVX", gv)


def generate_secondar_volume_datas(gv: int) -> Iterable[ObjectMetadata]:
    yield ObjectMetadata(SECONDAR_ROOT, f"CAV{gv}", "CAVX", gv)
    yield ObjectMetadata(SECONDAR_ROOT, f"ST{gv}_V001", "STX_V001", gv)
    for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        yield ObjectMetadata(SECONDAR_ROOT, f"RI{gv}_V{i:03d}", f"RIX_V{i:03d}", gv)


def generate_secondar_junction_datas(gv: int) -> Iterable[ObjectMetadata]:
    yield ObjectMetadata(SECONDAR_ROOT, f"ST{gv}_J001", "STX_J001", gv)
    yield ObjectMetadata(SECONDAR_ROOT, f"CAVST{gv}", "CAVSTX", gv)
    yield ObjectMetadata(SECONDAR_ROOT, f"CAVRI{gv}", "CAVRIX", gv)
    for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        yield ObjectMetadata(SECONDAR_ROOT, f"RI{gv}_J{i:03d}", f"RIX_J{i:03d}", gv)
    yield ObjectMetadata(SECONDAR_ROOT, f"RICAV{gv}", f"RICAVX", gv)


def generate_secondar_wall_datas(gv: int) -> Iterable[ObjectMetadata]:
    for i in (1, 2, 3):
        yield ObjectMetadata(SECONDAR_ROOT, f"WC{i}{gv}", f"WC{i}X", gv)
    yield ObjectMetadata(SECONDAR_ROOT, f"R1{gv}  001", "R1X  001", gv)
    yield ObjectMetadata(SECONDAR_ROOT, f"R2{gv}  001", "R2X  001", gv)
    for i in (1, 2, 3, 4, 5):
        for j in (3, 4):
            yield ObjectMetadata(SECONDAR_ROOT, f"R{j}{gv}  {i:03d}", f"R{j}X  {i:03d}", gv)
    for i in (1, 2, 3, 4, 5):
        for j in (5, 6):
            yield ObjectMetadata(SECONDAR_ROOT, f"R{j}{gv}  {i:03d}", f"R{j}X  {i:03d}", gv)
    yield ObjectMetadata(SECONDAR_ROOT, f"WS{gv}  001", f"WSX  1", gv)
    yield ObjectMetadata(SECONDAR_ROOT, f"R7{gv}  001", "R7X  001", gv)


def generate_primary_volume_datas(gv: int) -> Iterable[ObjectMetadata]:
    yield ObjectMetadata(PRIMARY_ROOT, f"{HOT_BOX_PREFIX}{gv}", "HBX", gv)
    for i in (1, 2, 3, 4, 5, 6, 7, 8):
        yield ObjectMetadata(PRIMARY_ROOT, f"TU{gv}_V{i:03d}", f"TUX_V{i:03d}", gv)
    yield ObjectMetadata(PRIMARY_ROOT, f"CB{gv}", "CBX", gv)


def generate_primary_wall_datas(gv: int) -> Iterable[ObjectMetadata]:
    yield ObjectMetadata(PRIMARY_ROOT, f"WHB{gv}", f"WHBX", gv)
    for (i, j) in ((1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1)):
        yield ObjectMetadata(PRIMARY_ROOT, f"T{i}{gv}  {j:03d}", f"T{i}X  {j:03d}", gv)
    yield ObjectMetadata(PRIMARY_ROOT, f"WCB{gv}", "WCBX", gv)


def generate_primary_junction_datas(gv: int) -> Iterable[ObjectMetadata]:
    yield ObjectMetadata(PRIMARY_ROOT, f"HBTU{gv}", "HBTUX", gv)
    yield ObjectMetadata(PRIMARY_ROOT, f"HLHB{gv}", "HLHBX", gv)
    yield ObjectMetadata(PRIMARY_ROOT, f"CBCL{gv}", "CBCLX", gv)
    for i in (1, 2, 3, 4, 5, 6, 7):
        yield ObjectMetadata(PRIMARY_ROOT, f"TU{gv}_J{i:03d}", f"TUX_J{i:03d}", gv)
    yield ObjectMetadata(PRIMARY_ROOT, f"TUCB{gv}", "TUCBX", gv)


def generate_sensor_paths(datas: Iterable[ObjectMetadata], mapping: NameMapping):
    
    from fluent.fluent_astec import ROOT

    for data in datas:
        index = mapping.sensor_indexes[data.name]
        yield ROOT.sensor(index).child_float("value", 1 - pyas.odessa_shift()), PathMetadata("value", data, True, False)


def generate_connecti_paths(datas: Iterable[ObjectMetadata], mapping: NameMapping):

    from fluent.fluent_astec import ROOT

    for data in datas:
        index = mapping.connecti_indexes[data.name]

        connecti = ROOT.connecti(index)

        yield connecti.child_str("STAT", 1 - pyas.odessa_shift()), PathMetadata("STAT", data, True, False)
        yield connecti.child_float("SECT", 1 - pyas.odessa_shift()), PathMetadata("STAT", data, True, False)
        flow = connecti.child_base("SOURCE", 1 - pyas.odessa_shift()).child_r1("FLOW", 1 - pyas.odessa_shift())
        yield flow.child_element(2 - pyas.odessa_shift()), PathMetadata("FLOW:MASS", data, True, True)
        yield flow.child_element(3 - pyas.odessa_shift()), PathMetadata("FLOW:ENERGY", data, True, True)


def generate_primary_volume_paths(datas: Iterable[ObjectMetadata], mapping: NameMapping):

    from fluent.fluent_astec import ROOT

    for data in datas:
        index = mapping.primary_volume_indexes[data.name]

        volume = ROOT.primary().volume(index)
        ther = volume.ther()
        yield ther.child_r1("P_h2", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("P_h2", data, True, True)
#        yield ther.child_r1("P_bho2", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("P_bho2", data, True, True)
#        yield ther.child_r1("P_co2", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("P_co2", data, True, True)
        yield ther.child_r1("P_steam", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("P_steam", data, True, True)
        yield ther.child_r1("T_gas", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("T_gas", data, True, True)
        yield ther.child_r1("T_liq", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("T_liq", data, True, True)
        yield ther.child_r1("x_alfa", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("x_alfa", data, True, True)


def generate_secondar_volume_paths(datas: Iterable[ObjectMetadata], mapping: NameMapping):

    from fluent.fluent_astec import ROOT

    for data in datas:
        index = mapping.secondar_volume_indexes[data.name]

        volume = ROOT.secondar().volume(index)
        ther = volume.ther()
#        yield ther.child_r1("P_h2", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("P_h2", data, True, True)
#        yield ther.child_r1("P_bho2", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("P_bho2", data, True, True)
#        yield ther.child_r1("P_co2", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("P_co2", data, True, True)
        yield ther.child_r1("P_steam", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("P_steam", data, True, True)
        yield ther.child_r1("T_gas", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("T_gas", data, True, True)
        yield ther.child_r1("T_liq", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("T_liq", data, True, True)
        yield ther.child_r1("x_alfa", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("x_alfa", data, True, True)
        if HOT_BOX_PREFIX in data.name:
            yield ther.child_r1("x_alfa1", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("x_alfa1", data, True, True)


def generate_primary_junction_paths(datas: Iterable[ObjectMetadata], mapping: NameMapping):

    from fluent.fluent_astec import ROOT

    for data in datas:
        index = mapping.primary_junction_indexes[data.name]

        junction = ROOT.primary().junction(index)

        yield junction.close(), PathMetadata("CLOSE", data, True, False)
        ther = junction.ther(1 - pyas.odessa_shift())
        yield ther.child_r1("v_gas", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("v_gas", data, True, True)
        yield ther.child_r1("v_liq", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("v_liq", data, True, True)


def generate_secondar_junction_paths(datas: Iterable[ObjectMetadata], mapping: NameMapping):

    from fluent.fluent_astec import ROOT

    for data in datas:
        index = mapping.secondar_junction_indexes[data.name]

        junction = ROOT.secondar().junction(index)
        yield junction.close(), PathMetadata("CLOSE", data, True, False)
        ther = junction.ther(1 - pyas.odessa_shift())

        yield ther.child_r1("v_gas", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("v_gas", data, True, True)
        yield ther.child_r1("v_liq", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata("v_liq", data, True, True)


def generate_primary_wall_paths(datas: Iterable[ObjectMetadata], mapping: NameMapping):

    from fluent.fluent_astec import ROOT

    primary = ROOT.primary()

    for data in datas:
        index = mapping.primary_wall_indexes[data.name]

        wall = primary.wall(index)
        for ther_index in [1 - pyas.odessa_shift(), 2 - pyas.odessa_shift()]:
            yield wall.ther(ther_index).child_r1("T_wall", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata(f"T_wall {ther_index}", data, True, True)
        yield wall.child_float("fp_power", 1 - pyas.odessa_shift()), PathMetadata("fp_power", data, True, False)


def generate_secondar_wall_paths(datas: Iterable[ObjectMetadata], mapping: NameMapping):

    from fluent.fluent_astec import ROOT

    for data in datas:
        index = mapping.secondar_wall_indexes[data.name]

        wall = ROOT.secondar().wall(index)

        for ther_index in [1 - pyas.odessa_shift(), 2 - pyas.odessa_shift()]:
            yield wall.ther(ther_index).child_r1("T_wall", 1 - pyas.odessa_shift(),).child_element(2 - pyas.odessa_shift()), PathMetadata(f"T_wall {ther_index}", data, True, True)
        yield wall.child_float("fp_power", 1 - pyas.odessa_shift()), PathMetadata("fp_power", data, True, False)


def generate_gv_paths(gv: int, mapping: NameMapping):
    yield from generate_secondar_volume_paths(generate_secondar_volume_datas(gv), mapping)
    yield from generate_secondar_junction_paths(generate_secondar_junction_datas(gv), mapping)
    yield from generate_secondar_wall_paths(generate_secondar_wall_datas(gv), mapping)

    yield from generate_primary_volume_paths(generate_primary_volume_datas(gv), mapping)
    yield from generate_primary_junction_paths(generate_primary_junction_datas(gv), mapping)
    yield from generate_primary_wall_paths(generate_primary_wall_datas(gv), mapping)

    yield from generate_connecti_paths(generate_connecti_datas(gv), mapping)
    yield from generate_sensor_paths(generate_sensor_datas(gv), mapping)


def generate_paths_and_metadatas(mapping: NameMapping):
    for gv in GVS:
        yield from generate_gv_paths(gv, mapping)


def generate_paths(mapping: NameMapping):
    for path, _ in generate_paths_and_metadatas(mapping):
        yield path


def generate_datas(gv: int):
    yield from generate_primary_volume_datas(gv)
    yield from generate_primary_wall_datas(gv)
    yield from generate_primary_junction_datas(gv)
    yield from generate_secondar_volume_datas(gv)
    yield from generate_secondar_wall_datas(gv)
    yield from generate_secondar_junction_datas(gv)
    yield from generate_connecti_datas(gv)
    yield from generate_sensor_datas(gv)


def gv_path_name_to_gv_name(gv: int, mapping: NameMapping):
    for path, metadata in generate_gv_paths(gv, mapping):
        yield path.odessa_repr, metadata.gv_name


def gv_to_path_name_to_gv_name(mapping: NameMapping):
    for gv in GVS:
        yield gv, dict(gv_path_name_to_gv_name(gv, mapping))


def get_gv_model_metadata(base):
    import pyastec as pyas
    from fluent.fluent_astec import ROOT

    primary = ROOT.primary()
    secondar = ROOT.secondar()

    sensor_names = {}
    connecti_names = {}
    primary_volume_names = {}
    secondar_volume_names = {}
    primary_junction_names = {}
    secondar_junction_names = {}
    primary_wall_names = {}
    secondar_wall_names = {}

    for index in range(1 - pyas.odessa_shift(), ROOT.size_from("CONNECTI", base) + 1 - pyas.odessa_shift()):
        connecti_names[index] = ROOT.connecti(index).name_from(base)

    for index in range(1 - pyas.odessa_shift(), ROOT.size_from("SENSOR", base) + 1 - pyas.odessa_shift()):
        sensor_names[index] = ROOT.sensor(index).name_from(base)

    for index in range(1 - pyas.odessa_shift(), primary.size_from("VOLUME", base) + 1 - pyas.odessa_shift()):
        primary_volume_names[index] = primary.volume(index).name_from(base)

    for index in range(1 - pyas.odessa_shift(), secondar.size_from("VOLUME", base) + 1 - pyas.odessa_shift()):
        secondar_volume_names[index] = secondar.volume(index).name_from(base)

    for index in range(1 - pyas.odessa_shift(), primary.size_from("JUNCTION", base) + 1 - pyas.odessa_shift()):
        primary_junction_names[index] = primary.junction(index).name_from(base)

    for index in range(1 - pyas.odessa_shift(), secondar.size_from("JUNCTION", base) + 1 - pyas.odessa_shift()):
        secondar_junction_names[index] = secondar.junction(
            index).name_from(base)

    for index in range(1 - pyas.odessa_shift(), primary.size_from("WALL", base) + 1 - pyas.odessa_shift()):
        primary_wall_names[index] = primary.wall(index).name_from(base)

    for index in range(1 - pyas.odessa_shift(), secondar.size_from("WALL", base) + 1 - pyas.odessa_shift()):
        secondar_wall_names[index] = secondar.wall(index).name_from(base)

    name_mapping = NameMapping(connecti_names=connecti_names,
                       secondar_volume_names=secondar_volume_names,
                       primary_volume_names=primary_volume_names,
                       primary_junction_names=primary_junction_names,
                       secondar_junction_names=secondar_junction_names,
                       primary_wall_names=primary_wall_names,
                       secondar_wall_names=secondar_wall_names,
                       sensor_names=sensor_names)

    metadata_list = []
    input_index_to_name = {}
    input_name_to_index = {}
    output_index_to_name = {}
    output_name_to_index = {}
    input_indexes = []
    output_indexes = []
    input_index = 0
    output_index = 0
    for (index, (_, metadata)) in enumerate(generate_gv_paths(GVS[0], name_mapping)):
        metadata_list.append(metadata)
        if metadata.input:
            input_index_to_name[input_index] = metadata.gv_name
            input_name_to_index[metadata.gv_name] = input_index
            input_indexes.append(index)
            input_index += 1
        if metadata.output:
            output_index_to_name[output_index] = metadata.gv_name
            output_name_to_index[metadata.gv_name] = output_index
            output_indexes.append(index)
            output_index += 1
    return GvModelMetadata(name_mapping=name_mapping,
                            metadatas=metadata_list,
                           input_index_to_name=input_index_to_name,
                           input_name_to_index=input_name_to_index,
                           output_index_to_name=output_index_to_name,
                           output_name_to_index=output_name_to_index,
                           input_indexes=input_indexes,
                           output_indexes=output_indexes)


def load_gv_model_metadata(file_name: str) -> GvModelMetadata:
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
        raise Exception(f"Name mapping doesn't exist in {file_name}")

def dump_gv_model_metadata(file_name: str, gv_model_metadata: GvModelMetadata):
    pickle.dump(gv_model_metadata, open(file_name, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)

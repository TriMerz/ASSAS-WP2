import pyastec as pyas
import sys
from fluent.fluent_astec import ROOT
import networkx as nx
import json


def primary_nodes_volume(base: pyas.odbase):
    primary = ROOT.primary()
    for index in range(primary.size_from("VOLUME", base)):
        yield primary.volume(index).name_from(base), {"index":index, "type":"VOLUME"}

def secondar_nodes_volume(base: pyas.odbase):
    secondar = ROOT.secondar()
    for index in range(secondar.size_from("VOLUME", base)):
        yield secondar.volume(index).name_from(base), {"index":index, "type":"VOLUME"}

def primary_nodes_junction(base: pyas.odbase):
    primary = ROOT.primary()
    for index in range(primary.size_from("JUNCTION", base)):
        yield primary.junction(index).name_from(base), {"index":index, "type":"JUNCTION"}

def secondar_nodes_junction(base: pyas.odbase):
    secondar = ROOT.secondar()
    for index in range(secondar.size_from("JUNCTION", base)):
        yield secondar.junction(index).name_from(base), {"index":index, "type":"JUNCTION"}

def primary_nodes_wall(base: pyas.odbase):
    primary = ROOT.primary()
    for index in range(primary.size_from("WALL", base)):
        yield primary.wall(index).name_from(base), {"index":index, "type":"WALL"}

def secondar_nodes_wall(base: pyas.odbase):
    secondar = ROOT.secondar()
    for index in range(secondar.size_from("WALL", base)):
        yield secondar.wall(index).name_from(base), {"index":index, "type":"WALL"}

def primary_nodes_connecti(base: pyas.odbase):

    size = ROOT.size_from("CONNECTI", base)

    for index in range(size):
        if "PRIMARY" in [ROOT.connecti(index).to().get_from(base), ROOT.connecti(index).from_().get_from(base)]:
            yield ROOT.connecti(index).name_from(base), {"index":index, "type":"CONNECTI"}

def secondar_nodes_connecti(base: pyas.odbase):

    size = ROOT.size_from("CONNECTI", base)

    for index in range(size):
        if "SECONDAR" in [ROOT.connecti(index).to().get_from(base), ROOT.connecti(index).from_().get_from(base)]:
            yield ROOT.connecti(index).name_from(base), {"index":index, "type":"CONNECTI"}

def primary_edge_volume_junction(base: pyas.odbase):
    primary = ROOT.primary()
    for junction_pos in range(primary.size_from("JUNCTION", base)):
        yield from edge_volume_junction(primary.junction(junction_pos), base)

def secondar_edge_volume_junction(base: pyas.odbase):
    secondar = ROOT.secondar()
    for junction_pos in range(secondar.size_from("JUNCTION", base)):
        yield from edge_volume_junction(secondar.junction(junction_pos), base)

def edge_volume_junction(junction_path, base: pyas.odbase):
    close = junction_path.close().get_from(base)
    name = junction_path.name_().get_from(base)
    color = "red" if close else "green"
    yield (junction_path.nv_up().get_from(base), name, {"label":"VOLUME-JUNCTION", "color":color})
    yield (name, junction_path.nv_down().get_from(base), {"label":"JUNCTION-VOLUME", "color":color})

def secondar_edge_volume_wall(base: pyas.odbase):
    secondar = ROOT.secondar()
    for wall_pos in range(secondar.size_from("WALL", base)):
        wall = secondar.wall(wall_pos)
        yield edge_volume_wall(wall, base)

def primary_edge_volume_wall(base: pyas.odbase):
    primary = ROOT.primary()
    for wall_pos in range(primary.size_from("WALL", base)):
        wall = primary.wall(wall_pos)
        yield edge_volume_wall(wall, base)

def edge_volume_wall(wall, base: pyas.odbase):
    return (wall.volume().get_from(base), wall.name_().get_from(base), {"label":"VOLUME-WALL"})

def primary_edge_volume_connecti(base: pyas.odbase):

    connecti_count = ROOT.size_from("CONNECTI", base)

    for connecti_index in range(connecti_count):
        connecti = ROOT.connecti(connecti_index)

        if "PRIMARY" in connecti.from_().get_from(base):
            
            name = connecti.name_().get_from(base)

            if name in ["F_PRESSU", "F_UPHEA"]:
                wall = connecti.child_str("WALL", 0).get_from(base)
                yield (wall, name, {"label":"WALL-CONNECTI"})
            else:
                yield from edge_volume_connecti_from(connecti, base)

    for connecti_index in range(connecti_count):
        connecti = ROOT.connecti(connecti_index)
        
        if "PRIMARY" in connecti.to().get_from(base):
            yield from edge_volume_connecti_to(connecti, base)

def secondar_edge_volume_connecti(base: pyas.odbase):

    connecti_count = ROOT.size_from("CONNECTI", base)

    for connecti_index in range(connecti_count):

        connecti = ROOT.connecti(connecti_index)
        
        if "SECONDAR" in connecti.from_().get_from(base):
            yield from edge_volume_connecti_from(connecti, base)

    for connecti_index in range(connecti_count):

        connecti = ROOT.connecti(connecti_index)
            
        if "SECONDAR" in connecti.to().get_from(base):

            if "PRIMARY" in connecti.from_().get_from(base):
                volume_index = 1
            else:
                volume_index = 0
            yield from edge_volume_connecti_to(connecti, base, volume_index)

def edge_volume_connecti_from(connecti, base: pyas.odbase):

    name = connecti.name_().get_from(base)
    
    volume = connecti.child_str("VOLUME", 0).get_from(base)

    yield (volume, name, {"label":"VOLUME-CONNECTI"})

def edge_volume_connecti_to(connecti, base: pyas.odbase, volume_index:int=0):

    name = connecti.name_().get_from(base)

    volume = connecti.child_str("VOLUME", volume_index).get_from(base)

    yield (name, volume, {"label":"CONNECTI-VOLUME"})

def primary_edges(base:pyas.odbase):
    yield from primary_edge_volume_wall(base)
    yield from primary_edge_volume_junction(base)
    yield from primary_edge_volume_connecti(base)

def secondar_edges(base:pyas.odbase):
    yield from secondar_edge_volume_wall(base)
    yield from secondar_edge_volume_junction(base)
    yield from secondar_edge_volume_connecti(base)

def primary_nodes(base:pyas.odbase):
    yield from primary_nodes_volume(base)
    yield from primary_nodes_wall(base)
    yield from primary_nodes_junction(base)
    yield from primary_nodes_connecti(base)

def secondar_nodes(base:pyas.odbase):
    yield from secondar_nodes_volume(base)
    yield from secondar_nodes_wall(base)
    yield from secondar_nodes_junction(base)
    yield from secondar_nodes_connecti(base)

def graph(nodes, edges, name):
    graph = nx.MultiDiGraph(name=name)
    
    for (node, node_attr) in nodes:
        graph.add_node(node, **node_attr)

    for (edge_from, edge_to, edge_attr) in edges:
        graph.add_edge(edge_from, edge_to, **edge_attr)

    return graph

def dump_graph(graph, output):
    nx.drawing.nx_pydot.write_dot(graph, f"{output}.dot")

    with open(f"{output}.json", 'w', encoding='utf-8') as f:
        json.dump(graph, f, default=nx.node_link_data)

if __name__ == "__main__":

    dir = sys.argv[1]
    time = float(sys.argv[2])
    output_primary = sys.argv[3]
    output_secondar = sys.argv[4]

    base = pyas.odloaddir(dir, time)

    primary = graph(primary_nodes(base), primary_edges(base), "primary")
    secondar = graph(secondar_nodes(base), secondar_edges(base), "secondary")
    
    dump_graph(primary, output_primary)
    dump_graph(secondar, output_secondar)

    
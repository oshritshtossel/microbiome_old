# from openpyxl import Workbook, load_workbook
import re
import math
import pandas
import networkx as nx
import pickle


"""
Every bacteria is an object to easily store its information
"""
class Bacteria:
    def __init__(self, string, val):
        string = string.replace(" ", "")
        lst = re.split(";|__", string)
        self.val = val
        # removing letters and blank spaces
        for i in range(0, len(lst)):
            if len(lst[i]) < 2:
                lst[i] = 0
        lst = [value for value in lst if value != 0]
        self.lst = lst

"""
    Creates the taxonomy tree
    series: the pandas series with the OTUs and correlations
    flag: a numberto remove from the tree, or None for no removal.
    keepFlagged: whether to remove from the tree or from the graph alltogether.
"""
def create_tax_tree(series, flag=None, keepFlagged=False):
    graph = nx.Graph()
    graph.add_node(("Bacteria",), val=0)
    graph.add_node(("Archaea",), val=0)
    bac = []
    for i, (tax, val) in enumerate(series.items()):
        # adding the bacteria in every column
        bac.append(Bacteria(tax, val))
        # connecting to the root of the tempGraph
        graph.add_edge(("Anaerobe",), (bac[i].lst[0],))
        # connecting all levels of the taxonomy
        for j in range(0, len(bac[i].lst) - 1):
            updateval(graph, bac[i], j, True)
        # adding the value of the last node in the chain
        updateval(graph, bac[i], len(bac[i].lst) - 1, False)
    graph.nodes[("Anaerobe",)]["val"] = graph.nodes[("Bacteria",)]['val']+graph.nodes[("Archaea",)]['val']
    return create_final_graph(graph, flag, keepFlagged)


def updateval(graph, bac, num, adde):
    # adding the nodes to the graph
    if adde:
        if tuple(bac.lst[:num+1]) not in graph:
            graph.add_node(tuple(bac.lst[:num+1]), val=0)
        if tuple(bac.lst[:num+2]) not in graph:
            graph.add_node(tuple(bac.lst[:num+2]), val=0)

    #adding the edge between two levels of taxonomy
        graph.add_edge(tuple(bac.lst[:num+1]), tuple(bac.lst[:num+2]))

    new_val = graph.nodes[tuple(bac.lst[:num+1])]['val'] + bac.val
    # set values
    graph.nodes[tuple(bac.lst[:num+1])]['val'] = new_val


"""
    Removes unwanted nodes and finalizes the graph.
"""
def create_final_graph(graph, flag, keepFlagged):
    # removes edges with a flagged node
    if flag is not None:
        for e in graph.edges():
            if (graph.nodes[e[0]]["val"] == flag or graph.nodes[e[1]]["val"] == flag):
                graph.remove_edge(*e)

    # removes flagged nodes all together
    if not keepFlagged:
        graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph

if __name__ == "__main__":
    create_tax_tree(pickle.load(open("graph152forAriel.p", "rb")), flag=0, keepFlagged=True)

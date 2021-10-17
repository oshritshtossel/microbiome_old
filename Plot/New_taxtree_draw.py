import networkx as nx
import pandas as pd
from LearningMethods.taxtreeCreate import create_tax_tree
from ete3 import Tree, NodeStyle, TreeStyle
import svgutils.compose as sc
from IPython.display import SVG # /!\ note the 'SVG' function also in svgutils.compose
import numpy as np
import matplotlib.pyplot as plt

def name_to_newick(tup):
    return str(tup).replace(", ", "|").replace("(", "<") \
        .replace(")", ">").replace("'", "").replace(",", "")


def newick_to_name(string):
    return tuple(string.strip("<>").split("|"))


def tree_to_newick_recursion(g, root=("anaerobe",)):
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    for child in g[root]:
        if len(child) > len(root) or root == ("anaerobe",):
            if len(g[child]) > 1:
                subgs.append(tree_to_newick_recursion(g, root=child))
            else:
                subgs.append(name_to_newick(child))
    return "(" + ','.join(subgs) + ")" + name_to_newick(root)


def tree_to_newick(s):
    graph = create_tax_tree(s)
    newick = tree_to_newick_recursion(graph) + ";"
    return newick, graph


def get_tree_shape(newick, graph, lower_threshold, higher_threshold, dict):
    t = Tree(newick, format=8)
    not_yellows = []
    for n in t.traverse():
        nstyle = NodeStyle()
        nstyle["fgcolor"] = dict["netural"]
        name = newick_to_name(n.name)
        if name != '':
            if graph.nodes[name]["val"] > higher_threshold:
                nstyle["fgcolor"] = dict["positive"]
                not_yellows.append(n)
            elif graph.nodes[name]["val"] < lower_threshold:
                nstyle["fgcolor"] = dict["negative"]
                not_yellows.append(n)
        nstyle["size"] = 5
        n.set_style(nstyle)
    d = 1
    while(d > 0):
        d = 0
        for n in t.traverse():
            if n.is_leaf():
                x = n.name
                flag = 1
                if nstyle["fgcolor"] != "yellow":
                    continue
                parent = n.up
                if parent == n.get_tree_root():
                    n.delete()
                while not parent.is_root():
                    if parent in not_yellows:
                        flag = 0
                        break
                    parent = parent.up
                if flag:
                    d+=1
                    n.delete()
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.min_leaf_separation = 0.5
    ts.mode = "c"
    ts.root_opening_factor = 0.75
    ts.show_branch_length = False
    return t, ts


def draw_tree(ax: plt.Axes ,series, dict):
    if type(dict["treshold"]) == tuple:
        lower_threshold, higher_threshold = dict["treshold"]
    else:
        lower_threshold, higher_threshold = -dict["treshold"], dict["treshold"]
    newick, graph = tree_to_newick(series)
    try:
        t, ts = get_tree_shape(newick, graph, lower_threshold, higher_threshold, dict)
    except:
        print("not enough bacterias to create a tree")
        return None
    for n in t.traverse():
        n.name = ";".join(n.name.strip("<>").split("|")[-2:]).replace("[","").replace("]","")
    t.render("phylotree.svg", tree_style=ts)
    t.render("phylotree.png", tree_style=ts)
    tree = plt.imread('./phylotree.png')
    im = ax.imshow(tree)
    plt.show()
    t.show(tree_style=ts)


if __name__ == "__main__":
    df = pd.read_csv("Plot/Data/stool_tax_6_log_sub_pca.csv", index_col=0)
    meta = pd.read_csv("Plot/Raw_data/metadata_parkinson_stool.csv", index_col =0)
    # make the tag binary
    meta["Patient"][meta["Patient"] == "P"] = 1
    meta["Patient"][meta["Patient"] == "C"] = 0
    tag = meta["Patient"]


    # s = df.iloc[0]
    df = df[df.columns[0]]
    draw_tree(df, 0.5)

import pickle
import networkx as nx
from seaborn.external.husl import rgb_to_hex
from LearningMethods.taxtreeCreate import create_tax_tree
from ete3 import Tree, NodeStyle, TreeStyle
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
d = {1: 'k', 2: 'p', 3: 'c', 4: 'o', 5: 'f', 6: 'g', 7: 's'}

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

def get_tax_str(tup):
    letters = ['k','p','c','o','f','g','s']
    i = 0
    str= ''
    while i < len(tup):
        str += letters[i] + '__' +tup[i]
        i+=1
        if i != len(tup):
            str += '; '
    return str

def tree_to_newick(s):
    graph = create_tax_tree(s)
    newick = tree_to_newick_recursion(graph) + ";"
    return newick, graph


def get_tree_shape(newick, graph, lower_threshold, higher_threshold, dict, apply=False, apply_purne=False):
    t = Tree(newick, format=8)
    not_yellows = []
    if apply == 'heat':
        values = dict['values']
        # min_v =  0
        # max_v = values[max(values, key=values.get)]
        cmap = mpl.cm.get_cmap(dict['colormap'])
        # for key in values:
        # values[key] = (values[key] - min_v) / (max_v - min_v)
        real_values = {}
        for v in values:
            real_values[delete_empty_taxonomic_levels(v)] = values[v]
        with open('../../../../Downloads/list_of_big_bact.pkl', 'rb') as f:
                common_in_tissue = pickle.load(f)
                common_in_tissue = [delete_empty_taxonomic_levels(l).replace(']', '').replace('[', '') for l in
                                     common_in_tissue]
    for n in t.traverse():
        nstyle = NodeStyle()
        nstyle["fgcolor"] = dict["netural"]
        name = newick_to_name(n.name)
        if name != '':
            if not apply and not n.is_root():
                if graph.nodes[name]["val"] > higher_threshold:
                    nstyle["fgcolor"] = dict["positive"]
                    not_yellows.append(n)
                elif graph.nodes[name]["val"] < lower_threshold:
                    nstyle["fgcolor"] = dict["negative"]
                    not_yellows.append(n)
            elif apply == 'heat':
                name = delete_suffix(delete_empty_taxonomic_levels(get_tax_str(name)))
                # f = delete_suffix(delete_empty_taxonomic_levels(get_tax_str(name).split(';')[]).replace('[','').replace(']','').replace(' ', ''))
                if delete_suffix(name) in common_in_tissue:
                       nstyle["size"] = 12
                       n.set_style(nstyle)
                else:
                    nstyle["size"] = 5
                # print(delete_suffix(delete_empty_taxonomic_levels(get_tax_str(name))))
                if name in real_values:
                    nstyle["fgcolor"] = rgb_to_hex(cmap(real_values[delete_suffix(name)])[:3])
                else:
                    nstyle["fgcolor"] = 'black'
                n.set_style(nstyle)
                not_yellows.append(n)
            else:
                pass
        if apply != 'heat':
            nstyle["size"] = 5
        n.set_style(nstyle)
    if apply_purne:
        d = 1
        while (d > 0):
            d = 0
            for n in t.traverse():
                if n.is_leaf():
                    flag = 1
                    if nstyle["fgcolor"] != dict['netural']:
                        continue
                    parent = n.up
                    if parent == n.get_tree_root():
                        n.delete()
                        continue
                    if parent.up == n.get_tree_root():
                        n.delete()
                        continue
                    if parent.up.up == n.get_tree_root():
                        n.delete()
                        continue
                    while not parent.up.up.is_root():
                        if parent in not_yellows:
                            flag = 0
                            break
                        parent = parent.up
                    if flag:
                        d += 1
                        n.delete()
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.min_leaf_separation = 0.5
    ts.mode = "c"
    ts.root_opening_factor = 0.75
    ts.show_branch_length = False
    return t,ts


def draw_tree(ax: plt.Axes, series, dict, apply_purne=True):
    # series = dict['list']
    if type(dict["treshold"]) == tuple:
        lower_threshold, higher_threshold = dict["treshold"]
    else:
        lower_threshold, higher_threshold = -dict["treshold"], dict["treshold"]
    newick, graph = tree_to_newick(series)
    try:
        t, ts = get_tree_shape(newick, graph, lower_threshold, higher_threshold, dict, apply_purne=apply_purne)
    except:
        print("not enough bacterias to create a tree")
        return None
    for n in t.traverse():
        while re.match(r'_+\d', n.name.split("|")[-1]):
            n.name = "<" + '|'.join(n.name.split('|')[:-1]) + ">"
        c = n.name.count('|') + 1
        n.name = ";".join(n.name.strip("<>").split("|")[-1:]).replace("[", "").replace("]", "")
        n.name += "   (" + str(d[c]) + ")"
        # n.name = delete_suffix(str(n.name))
    t.render("phylotree.svg", tree_style=ts)
    t.render("phylotree.png", tree_style=ts)
    tree = plt.imread('./phylotree.png')
    im = ax.imshow(tree)
    return 1


def delete_suffix(i):
    m = re.search(r'_+\d+$', i)
    if m is not None:
        i = i[:-(m.end() - m.start())]
    return i

def delete_empty_taxonomic_levels(i):
    splited = i.split(';')
    try:
        while re.search(r'[a-z]_+\d*$', splited[-1]) is not None:
            splited = splited[:-1]
    except:
        pass
    i = ""
    for j in splited:
        i += j
        i += ';'
    i = i[:-1]
    return i

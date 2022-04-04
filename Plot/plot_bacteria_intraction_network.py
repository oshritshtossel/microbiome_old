import pickle

import pandas as pd
from imgkit import IMGKit
from pyvis.network import Network
from pylab import *
import os
import seaborn as sns
from matplotlib.patches import Patch
import imgkit
from folium import utilities
from pyppeteer import launch

def shorten_single_bact_name(bacteria):
    bacteria = delete_suffix(bacteria)
    return  bacteria.split(';')[-1]

def delete_empty_taxonomic_levels(i):
    splited = i.split(';')
    while re.search(r'[a-z]_+\d*$', splited[-1]) is not None:
        splited = splited[:-1]
    i = ""
    for j in splited:
        i += j
        i += ';'
    i = i[:-1]
    print(i)
    return i


def delete_suffix(i):
    m = re.search(r'_+\d+$', i)
    if m is not None:
        i = i[:-(m.end() - m.start())]
    return i


def rgbA_colors_generator():
    r = sample(1)
    g = sample(1)
    b = sample(1)
    return (r[0], g[0], b[0])


def plot_table(bacterias, edge_list, edges_colors, nodes_with_edges_out, nodes_with_edges_in, short_names):

    colors = np.unique(np.array(edges_colors))
    d = {}
    d[0] = '#FFFFFF'
    i = 1
    for c in colors:
        d[str(i)] = c
        i+=1
    d = {v: k for k, v in d.items()}

    # init the edges matrix
    edges = []
    for i in range(len(bacterias)):
        edges.append([])
        for j in range(len(bacterias)):
            edges[i].append('#FFFFFF')

    # adding the edges
    i = 0
    for edge in edge_list:
        edges[edge[0]][edge[1]] = edges_colors[i]
        i+=1

    edges_final = []
    rows = []
    cols = []
    c = 0
    numbers_dict = {}
    for i in range(len(bacterias)):
        if i in nodes_with_edges_out:
            edges_final.append([])
            if short_names[i] not in numbers_dict:
                numbers_dict[short_names[i]] = c
                c+=1
            rows.append(short_names[i] + '  (' + str(numbers_dict[short_names[i]])+')')
            for j in range(len(bacterias)):
                if j in nodes_with_edges_in:
                    edges_final[-1].append(d[edges[i][j]])
    edges_final = np.array(edges_final)
    for j in range(len(bacterias)):
        if j in nodes_with_edges_in:
            if short_names[j] not in numbers_dict:
                numbers_dict[short_names[j]] = c
                c+=1
            cols.append(short_names[j] + '  (' + str(numbers_dict[short_names[j]])+')')
    pallete = []
    pallete.append('#FFFFFF')
    for c in colors:
        pallete.append(c)
    print(cols)
    g = sns.clustermap(data=edges_final, cmap=sns.color_palette(pallete),  vmin=0.0, vmax=len(pallete), linewidths=0.1, linecolor='gray', xticklabels=cols, yticklabels=rows, cbar_pos=None)
    lut = {'positive_corr':'blue', 'negative_corr':'red'}
    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut, title='intervations',
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    # plt.xticks(rotation=45)  # Rotates X-Axis Ticks by 45-degrees

    g.savefig('interaction.png', dpi=600)
    return  numbers_dict




# color must be in HEX!!!
def get_nodes_colors_by_bacteria_tax_level(bacteria, G_name, taxnomy_level, folder):
    bacteria = [b.split(";") for b in bacteria]
    s = pd.Series(bacteria)
    # get the taxonomy type for the wanted level
    taxonomy_reduced = s.map(lambda x: ';'.join(x[:taxnomy_level]))
    tax_groups = list(set(taxonomy_reduced))  # get the unique taxonomy types
    tax_groups.sort()  # sort for consistency of the color + shape of a group in multiple runs (who has the same groups)
    number_of_tax_groups = len(tax_groups)

    tax_to_color_and_shape_map = {}
    colors = ['#CD1414', '#EE831E', '#F0E31E', '#91F01E', '#1EF08D', '#1EDAF0', '#1E4EF0', '#671EF0', '#F01EE6', '#F01E5A',
              '#773326', '#695B18', '#787A6E', '#040404', '#1C440E', '#ABE0D6', '#ABBDE0', '#DCBFEC', '#ECBFD7', '#0400FF']
    markers = ['D', 'o', '*', '^', 'v']
    shapes = ["dot","diamond", "star", "triangle", "triangleDown"]
    shape_to_marker_map = {shapes[i]: markers[i] for i in range(len(markers))}

    for i in range(number_of_tax_groups):
        # c = plt_clr.rgb2hex(rgbA_colors_generator())  # random color
        tax_to_color_and_shape_map[tax_groups[i]] = (colors[i%len(colors)], shapes[0])  # color + shape from list

    color_list = [tax_to_color_and_shape_map[t][0] for t in taxonomy_reduced]
    shape_list = [tax_to_color_and_shape_map[t][1] for t in taxonomy_reduced]
    group_list = [shorten_single_bact_name(t) for t in taxonomy_reduced]

    # create the legend useing matplotlib because pyvis doesn't have legends
    # lines = []
    # for (key, val) in tax_to_color_and_shape_map.items():
    #     color = val[0]
    #     marker = shape_to_marker_map[val[1]]
    #     line = mlines.Line2D([], [], color=color, marker=marker,
    #                               markersize=15, label=key)
    #     lines.append(line)
    # plt.legend(handles=lines, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad = 0.)
    # plt.savefig(os.path.join(folder, G_name + "_legend.svg"), bbox_inches='tight', format='svg')
    # plt.show()
    return color_list, shape_list, group_list, tax_to_color_and_shape_map



def plot_bacteria_intraction_network(bacteria, node_list, node_size_list, edge_list, color_list, G_name, folder,
                                     color_by_tax_level=2, directed_G=True, control_color_and_shape=True):
    # create the results folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    nodes_colors, nodes_shapes, group_list, tax_to_color_and_shape_map = \
        get_nodes_colors_by_bacteria_tax_level(bacteria, G_name, taxnomy_level=color_by_tax_level, folder=folder)

    bact_short_names = [shorten_single_bact_name(b) for b in bacteria]
    nodes_with_edges_out = np.unique(np.array(edge_list).flatten()[::2]).tolist()
    nodes_with_edges_in = np.unique(np.array(edge_list).flatten()[1::2]).tolist()

    numbers_dict = plot_table(bacteria, edge_list, color_list, nodes_with_edges_out, nodes_with_edges_in, bact_short_names)


    net = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=directed_G)
    #net.barnes_hut(gravity=-120000)
    net.force_atlas_2based()

    # for the nodes: you can use either only the group option the automatically colors the group in different colors
    # shaped like dots - no control, or use color and shape to make it you own, in this case group is irrelevant
    for i, node in enumerate(node_list):
        if node in set().union(*[nodes_with_edges_out,nodes_with_edges_in]):
            if control_color_and_shape:
                net.add_node(int(node), label=str(numbers_dict[bact_short_names[i]]), color=nodes_colors[i], value=node_size_list[i],
                            shape='circle', group=group_list[i])
            else:
                net.add_node(int(node), label=str(numbers_dict[bact_short_names[i]]),  value=node_size_list[i],
                            group='circle')

    # for the edges, the colors are what you decide and send
    for i, (u, v) in enumerate(edge_list):
        net.add_edge(u, v, color=color_list[i])

    t_name = 'interaction.png'
    # plt.savefig(t_name)
    net.save_graph(os.path.join(folder, G_name + ".html"))
    # net.show(G_name + ".html")
    # options = {
    #     'format': 'png',
    # }
    # hti = Html2Image(browser='chrome')
    # hti.screenshot(html_file=G_name + '.html', save_as=G_name + '.png')
    return  G_name + '.png', t_name

#2d arr of correlations
def get_data(arr, treshold_min=-1, treshold_max=1):
    node_list = list(np.arange(arr.shape[0]))
    edge_list = []
    color_list = []
    arr[(arr <treshold_max) & (arr > treshold_min)] = 0
    arr[arr >=treshold_max] = 1
    arr[arr<=treshold_min] = 2
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i,j] != 0:
                edge_list.append((i,j))
                if arr[i,j] == 1:
                    color_list.append('#0000FF')
                else:
                    color_list.append('#FF0000')
    return  node_list, edge_list, color_list


def main(path,name):
    df = pd.read_csv(path)
    df.index = df.iloc[:,0]
    to_append = []
    for id in df.index:
        if id not in df.columns:
            to_append.append(id)
    for c in to_append:
        df[c] = np.zeros(len(df.index))
    df = df.iloc[:,1:]
    values = np.sort(df.to_numpy().flatten())
    arr = df.to_numpy()
    for i in range(arr.shape[0]):
        arr[i,i] = 0
    node_list, edge_list, color_list = get_data(arr, treshold_min=min(values[int(0.004*len(values))],0), treshold_max=max(values[int(0.996*len(values))],0))

    bacteria = list(df.columns)
    # set the size of the nodes, can control it  if wanted
    v = [100] * len(bacteria)

    t_name, g_name = plot_bacteria_intraction_network(bacteria, node_list, v, edge_list, color_list, "example_graph", "bacteria_interaction_network")
    #############################################################
    g_name = 'bacteria_interaction_network/example_graph.html'
    t_name = 'interaction.png'
    imgkit.from_file(g_name, 'example_graph.png')
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(18.5, 10.5)
    t_name  = plt.imread(t_name)
    g_name = plt.imread( 'example_graph.png')
    axes[0].imshow(g_name)
    axes[1].imshow(t_name)
    plt.subplots_adjust(top=1.0,bottom=0.0,left=0.035,right=0.975,hspace=0.1,wspace=0.07)
    plt.show()
    fig.savefig(f'{name}.pdf', dpi=500)


if __name__ == '__main__':
    ## squared matriix of the coeefients
    main('../../PycharmProjects/saliva/coeff_lasso.csv', name='saliva_all')
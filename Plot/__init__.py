with open('../../PycharmProjects/saliva/intersection_edges.pickle', 'rb') as handle:
    edges_intersection = pickle.load(handle)
edge_list = []
color_list = []
for tup in edges_intersection[1]:
    arr[list(df.columns).index(tup[0]), list(df.columns).index(tup[1])] = 1
    edge_list.append((list(df.columns).index(tup[0]), list(df.columns).index(tup[1])))
    color_list.append('#0000FF')
for tup in edges_intersection[2]:
    arr[list(df.columns).index(tup[0]), list(df.columns).index(tup[1])] = 2
    edge_list.append((list(df.columns).index(tup[0]), list(df.columns).index(tup[1])))
    color_list.append('#FF0000')
node_list = list(np.arange(97))
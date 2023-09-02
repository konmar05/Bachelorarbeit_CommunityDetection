import pickle
import json
import networkx as nx
import networkx.algorithms.community.quality as m
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cdlib import algorithms, viz, evaluation, benchmark, datasets


def generate_graph_from_edgelist(filepointer):
    """
    reads an edge list and creates a complete networkX Graph
    file has to be in following form:
    >> 1 3
    >> 3 6
    each line is represents an edge, between to nodes, nodes have to be seperated by a whitespace
    :param filepointer: to read from the given file
    :return: networkX graph
    """
    list_edges = []
    lines = filepointer.readlines()
    for l in lines:
        tmp = l.strip()
        pair = tmp.split()
        list_edges.append(pair)

    graph = nx.Graph()
    for pair in list_edges:
        graph.add_edge(pair[0], pair[1])

    return graph


def format_data(data, indent=0):
    """
    function to format and style data in a JSON file
    :param data: dictionary to store in JSON file
    :param indent:
    :return: formatted data with indent according to depth
    """
    result = ""
    for key, value in data.items():
        result += " " * indent  # Einrückung entsprechend der Tiefe
        result += f'"{key}": '
        if isinstance(value, dict):
            result += "{\n" + format_data(value, indent + 4) + " " * indent + "}\n"
        else:
            result += f'"{value}"\n'
    return result


def show_fitness_scores(graph, node_clustering_obj):
    """
    functions for having a quick view in cmd-line to the major fitness scores from an algorithm
    :param graph: networkX graph
    :param node_clustering_obj: communities as NodeClustering object
    :return: no return parameter
    """
    print('Erkannte Communities: ', len(node_clustering_obj.communities))
    print('Modularity-Score: ', m.modularity(graph, node_clustering_obj.communities))
    print('avg_distance: ', evaluation.avg_distance(graph, node_clustering_obj))
    print('avg_embeddedness: ', evaluation.avg_embeddedness(graph, node_clustering_obj))
    print('avg_internal_degree: ', evaluation.average_internal_degree(graph, node_clustering_obj))
    print('edges_inside: ', evaluation.edges_inside(graph, node_clustering_obj))
    print('internal_edge_density: ', evaluation.internal_edge_density(graph, node_clustering_obj))
    print('size: ', evaluation.size(graph, node_clustering_obj))
    print('hub_dominance: ', evaluation.hub_dominance(graph, node_clustering_obj))
    print('scaled_density: ', evaluation.scaled_density(graph, node_clustering_obj))
    print('\n')


def show_modularity_scores(graph, node_clustering_obj):
    """
    functions for having q quick vieww in cmd-line to the modularity scores
    :param graph: networkX graph
    :param node_clustering_obj: communities as NodeClustering object
    :return: no return parameter
    """
    print('erdos_renyi: ', evaluation.erdos_renyi_modularity(graph, node_clustering_obj))
    print('link: ', evaluation.link_modularity(graph, node_clustering_obj))
    print('modularity_density: ', evaluation.modularity_density(graph, node_clustering_obj))
    print('modularity_overlap: ', evaluation.modularity_overlap(graph, node_clustering_obj))
    print('newman_girvan: ', evaluation.newman_girvan_modularity(graph, node_clustering_obj))
    print('z_modularity: ', evaluation.z_modularity(graph, node_clustering_obj))


def generate_benchmark_graphs():
    """
    function was used once, no further usage needed
    """
    grp_1, com1 = benchmark.GRP(n=50, s=13, v=13, p_in=0.5, p_out=0.05)
    grp_2, com2 = benchmark.GRP(n=50,  s=7, v=7, p_in=0.5, p_out=0.05)

    grp_3, com3 = benchmark.GRP(n=88, s=13, v=13, p_in=0.4, p_out=0.1)
    grp_4, com4 = benchmark.GRP(n=88, s=5, v=5, p_in=0.4, p_out=0.1)

    grp_5, com5 = benchmark.GRP(n=125, s=13, v=13, p_in=0.6, p_out=0.05)
    grp_6, com6 = benchmark.GRP(n=125, s=5, v=5, p_in=0.6, p_out=0.05)

    nx.write_edgelist(grp_1, 'benchmark/grp_1.txt', delimiter=' ')
    nx.write_edgelist(grp_2, 'benchmark/grp_2.txt', delimiter=' ')
    nx.write_edgelist(grp_3, 'benchmark/grp_3.txt', delimiter=' ')
    nx.write_edgelist(grp_4, 'benchmark/grp_4.txt', delimiter=' ')
    nx.write_edgelist(grp_5, 'benchmark/grp_5.txt', delimiter=' ')
    nx.write_edgelist(grp_6, 'benchmark/grp_6.txt', delimiter=' ')


def create_graph_layouts():
    """
    function was used one time for each graph, no further usage needed
    """
    for i in range(1, 7):
        # load and create graph
        fp1 = open('benchmark/grp_' + str(i) + '.txt', 'r')
        g = generate_graph_from_edgelist(fp1)
        fp1.close()

        # choose place to store .pkl file
        with open('benchmark/layouts/grp_' + str(i) + '.pkl', 'wb') as fp2:
            pickle.dump(nx.spring_layout(g), fp2)
        fp2.close()


def get_layout(graph):
    """

    :param graph: choose between graph 1, 2, 3, 4, 5, 6
    :return: dictionary with positions for nodes from networkX graph
    """
    file_url = 'benchmark/layouts/grp_' + str(graph) + '.pkl'
    with open(file_url, 'rb') as fp:
        dict_pos = pickle.load(fp)

    return dict_pos


def get_benchmark_graphs(graph):
    """

    :param graph: choose between graph 1, 2, 3, 4, 5, 6
    :return: networkX graph
    """
    file_url = 'benchmark/grp_' + str(graph) + '.txt'
    fp = open(file_url, 'r')
    g = generate_graph_from_edgelist(fp)
    fp.close()

    return g


def average_degree(graph):
    tmp = nx.degree(graph)
    mean = 0
    for i in tmp:
        mean += i[1]

    return mean / len(tmp)


def average_centrality(graph):
    tmp = nx.degree_centrality(graph)
    mean = sum(tmp.values()) / len(tmp)
    return mean


def get_graph_stats():
    """

    :return: dictionary with all stats from all benchmark graphs
    """
    data = {}

    for i in range(1, 7):
        g = get_benchmark_graphs(i)
        stats = {'nodes': nx.number_of_nodes(g),
                 'edges': nx.number_of_edges(g),
                 'density': nx.density(g),
                 'avg_shortest_path': nx.average_shortest_path_length(g),
                 'avg_degree': average_degree(g),
                 'avg_centrality': average_centrality(g),
                 #'avg_transitivity': evaluation.avg_transitivity(g, nx.nodes(g))
                 }

        name = 'Graph ' + str(i)
        data[name] = stats

    return data


def get_fitness_scores(graph, node_clustering_obj):
    """

    :param graph: networkX graph
    :param node_clustering_obj: NodeClustering object for communities
    :return: dictionary with all fitness scores
    """
    scores = {'communities': len(node_clustering_obj.communities),
              'avg_distance': evaluation.avg_distance(graph, node_clustering_obj),
              'avg_embeddedness': evaluation.avg_embeddedness(graph, node_clustering_obj),
              'avg_internal_degree': evaluation.average_internal_degree(graph, node_clustering_obj),
              'edges_inside': evaluation.edges_inside(graph, node_clustering_obj),
              'expansion': evaluation.expansion(graph, node_clustering_obj),
              'internal_edge_density': evaluation.internal_edge_density(graph, node_clustering_obj),
              'scaled_density': evaluation.scaled_density(graph, node_clustering_obj),
              'avg_transitivity': evaluation.avg_transitivity(graph, node_clustering_obj),
              'size': evaluation.size(graph, node_clustering_obj)}

    return scores


def get_modularity_scores(graph, node_clustering_obj):
    """

    :param graph: networkX graph
    :param node_clustering_obj: communities as NodeClustering object
    :return: dictionary with all modularity scores
    """
    scores = {'erdos_renyi': evaluation.erdos_renyi_modularity(graph, node_clustering_obj),
              'link': evaluation.link_modularity(graph, node_clustering_obj),
              'modularity_density': evaluation.modularity_density(graph, node_clustering_obj),
              # 'modularity_overlap': evaluation.modularity_overlap(graph, node_clustering_obj),
              'girvan_newman': evaluation.newman_girvan_modularity(graph, node_clustering_obj),
              'z_modularity': evaluation.z_modularity(graph, node_clustering_obj)}

    return scores


def get_scores(graph, node_clustering_obj):
    """

    :param graph: networkX graph
    :param node_clustering_obj: community nodes as NodeClustering object
    :return: a nested dictionary with all modularity and fitness scores calculated from one algorithm
    """
    data = {'communities': node_clustering_obj.communities,
            'modularity_scores': get_modularity_scores(graph, node_clustering_obj),
            'fitness_scores': get_fitness_scores(graph, node_clustering_obj)}

    return data


def get_data(file_url, graph=None, run=None, algorithm=None, type_of_score=None, score=None):

    with open(file_url, 'r') as file:
        tmp = json.load(file)

    if graph is None and run is None and algorithm is None and type_of_score is None and score is None:
        return tmp
    elif run is None and algorithm is None and type_of_score is None and score is None:
        return tmp[graph]
    elif algorithm is None and type_of_score is None and score is None:
        return tmp[graph][run]
    elif type_of_score is None and score is None:
        return tmp[graph][run][algorithm]
    elif score is None:
        return tmp[graph][run][algorithm][type_of_score]
    else:
        return tmp[graph][run][algorithm][type_of_score][score][2]


def write_graph_stats_to_file():

    tmp = get_graph_stats()

    with open('evaluation/graph_stats.json', 'w') as file:
        json.dump(tmp, file,  indent=4)


def test_subplotting():

    with open('evaluation/graph_stats.json', 'r') as file:
        tmp = json.load(file)

    x_axes = ['Graph 1', 'Graph 2', 'Graph 3', 'Graph 4', 'Graph 5', 'Graph 6']
    y_nodes = []
    y_edges = []
    y_density = []
    y_shortest_path = []
    y_degree = []

    for g in tmp:
        a = tmp[g]
        y_nodes.append(a['nodes'])
        y_edges.append(a['edges'])
        y_density.append(a['density'])
        y_shortest_path.append(a['avg_shortest_path'])
        y_degree.append(a['avg_degree'])

    plt.figure(figsize=(20, 12))
    grid = gridspec.GridSpec(3, 3)

    # local var for styling plots
    ls = ':'
    lw = 1

    axes_1 = plt.subplot(grid[0, :])
    axes_1.set_title('nodes')
    axes_1.scatter(x_axes, y_nodes)
    axes_1.axhline(min(y_nodes), c='green', ls=ls, lw=lw)
    axes_1.axhline((min(y_nodes) + max(y_nodes))/2, c='orange', ls=ls, lw=lw)
    axes_1.axhline(max(y_nodes), c='red', ls=ls, lw=lw)
    plt.ylim([min(y_nodes)-10, max(y_nodes)+10])

    axes_2 = plt.subplot(grid[1, :-1])
    axes_2.set_title('edges')
    axes_2.scatter(x_axes, y_edges)
    for i in y_edges:
        axes_2.axhline(i, c='blue', ls=ls, lw=lw)
    plt.ylim([min(y_edges)-50, max(y_edges)+50])

    axes_3 = plt.subplot(grid[1:, -1])
    axes_3.set_title('density')
    axes_3.scatter(x_axes, y_density)
    for i in y_density:
        axes_3.axhline(i, c='blue', ls=ls, lw=lw)
    plt.ylim([min(y_density)-0.01, max(y_density)+0.01])

    axes_4 = plt.subplot(grid[-1, 0])
    axes_4.scatter(x_axes, y_degree)
    axes_4.set_title('degree')
    for i in y_degree:
        axes_4.axhline(i, c='blue', ls=ls, lw=lw)
    plt.ylim([min(y_degree)-1, max(y_degree)+1])

    axes_5 = plt.subplot(grid[-1, -2])
    axes_5.scatter(x_axes, y_shortest_path)
    axes_5.set_title('shortest_path')
    for i in y_shortest_path:
        axes_5.axhline(i, c='blue', ls=ls, lw=lw)
    plt.ylim([min(y_shortest_path)-0.1, max(y_shortest_path)+0.1])

    plt.tight_layout()
    plt.show()


def plot_modularity_scores():

    with open('evaluation/fitness_scores.json', 'r') as file:
        tmp = json.load(file)

    x_axes = ['erdos_renyi', 'link', 'modularity_density', 'modularity_overlap', 'girvan_newman', 'z_modularity']
    l_algorithms = ['louvain', 'lable_propagation', 'random_walk', 'eigenvector', 'belief', 'infomap']

    y_erdos = []
    y_link = []
    y_m_density = []
    y_m_overlap = []
    y_gir_new = []
    y_z_mod = []

    for algo in l_algorithms:
        y_erdos.append(get_data('graph_1', algo, 'modularity_scores', 'erdos_renyi'))

    for algo in l_algorithms:
        y_link.append(get_data('graph_1', algo, 'modularity_scores', 'link'))

    for algo in l_algorithms:
        y_m_density.append(get_data('graph_1', algo, 'modularity_scores', 'modularity_density'))

    for algo in l_algorithms:
        y_m_overlap.append(get_data('graph_1', algo, 'modularity_scores', 'modularity_overlap'))

    for algo in l_algorithms:
        y_gir_new.append(get_data('graph_1', algo, 'modularity_scores', 'girvan_newman'))

    for algo in l_algorithms:
        y_z_mod.append(get_data('graph_1', algo, 'modularity_scores', 'z_modularity'))

    plt.figure(figsize=(20, 12))

    plt.scatter(x_axes, y_erdos)
    plt.scatter(x_axes, y_link)
    plt.scatter(x_axes, y_m_density)
    plt.scatter(x_axes, y_m_overlap)
    plt.scatter(x_axes, y_gir_new)
    plt.scatter(x_axes, y_z_mod)
    plt.yscale('log')

    plt.show()


def plot_modularity_over_runs(graph, score):

    x_axes = []
    y_axes = []
    for i in range(1, 21):
        x_axes.append('run_' + str(i))
        y_axes.append(get_data('evaluation/second_test.json', graph, 'run_' + str(i), 'girvan_newman', 'modularity_scores', score))

    plt.scatter(x_axes, y_axes)
    plt.show()

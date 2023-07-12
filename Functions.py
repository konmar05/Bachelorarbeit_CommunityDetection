import pickle
import networkx as nx
import networkx.algorithms.community.quality as m
import matplotlib.pyplot as plt
from cdlib import algorithms, viz, evaluation, benchmark, datasets


def generate_graph_from_edgelist(filepointer):
    """
    reads an edge list and creates a complete networkX Graph
    file has to be in folling form:
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


def show_fitness_scores(graph, com_result):
    """
    functions for having a quick view in cmd-line to the major fitness scores from an algorithm
    :param graph: networkX graph
    :param com_result: communities as NodeClustering
    :return: no return parameter
    """
    print('Erkannte Communities: ', len(com_result.communities))
    print('Modularity-Score: ', m.modularity(graph, com_result.communities))
    print('avg_distance: ', evaluation.avg_distance(graph, com_result))
    print('avg_embeddedness: ', evaluation.avg_embeddedness(graph, com_result))
    print('avg_internal_degree: ', evaluation.average_internal_degree(graph, com_result))
    print('edges_inside: ', evaluation.edges_inside(graph, com_result))
    print('internal_edge_density: ', evaluation.internal_edge_density(graph, com_result))
    print('size: ', evaluation.size(graph, com_result))
    print('hub_dominance: ', evaluation.hub_dominance(graph, com_result))
    print('scaled_density: ', evaluation.scaled_density(graph, com_result))
    print('\n')


def show_modularity_scores(graph, com_result):
    """
    functions for having q quick vieww in cmd-line to the modularity scores
    :param graph: networkX graph
    :param com_result: communities as NodeClustering
    :return: no return parameter
    """
    print('erdos_renyi: ', evaluation.erdos_renyi_modularity(graph, com_result))
    print('link: ', evaluation.link_modularity(graph, com_result))
    print('modularity_density: ', evaluation.modularity_density(graph, com_result))
    print('modulaity_overlap: ', evaluation.modularity_overlap(graph, com_result))
    print('newman_girvan: ', evaluation.newman_girvan_modularity(graph, com_result))
    print('z_modularity: ', evaluation.z_modularity(graph, com_result))


def generate_benchmark_graphs():
    """
    function was used once, no further usage needed
    """
    grp_1, com1 = benchmark.GRP(55, 13, 5, 0.5, 0.05)
    grp_2, com2 = benchmark.GRP(55,  7, 5, 0.5, 0.05)
    grp_3, com3 = benchmark.GRP(55, 13, 8, 0.5, 0.05)

    grp_4, com4 = benchmark.GRP(81, 13, 8, 0.6, 0.075)
    grp_5, com5 = benchmark.GRP(81, 23, 8, 0.6, 0.075)
    grp_6, com6 = benchmark.GRP(81, 13, 14, 0.6, 0.075)

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
    for i in range(0, 6):
        tmp = str(i + 1)
        # load and create graph
        fp1 = open('benchmark/grp_' + tmp + '.txt', 'r')
        g = generate_graph_from_edgelist(fp1)
        fp1.close()

        # choose place to store .pkl file
        with open('benchmark/layouts/grp_' + tmp +'.pkl', 'wb') as fp2:
            pickle.dump(nx.spring_layout(g), fp2)
        fp2.close()


# todo
def create_bar_diagramm():

    plt.figure(figsize=(15, 7))
    # Paare von Datenwerten
    x_values = []
    y_values = []

    fp = open('testdaten/tmp.txt', 'r')
    list_pairs = []
    pairs = fp.readlines()

    for l in pairs:
        tmp = l.strip()
        pair = tmp.split()
        list_pairs.append(pair)

    for p in list_pairs:
        x_values.append(p[0])
        y_values.append(p[1])

    # Erstelle ein Balkendiagramm
    plt.bar(x_values, y_values)

    # Optionale Anpassungen des Diagramms
    plt.xlabel('X-Werte')
    plt.ylabel('Y-Werte')
    plt.title('Balkendiagramm der Datenpaare')

    # Zeige das Balkendiagramm an
    plt.show()


def get_layout(graph):
    """
    Attention, parameters have to be a string
    :param graph choose between graph 1, 2, 3, 4, 5, 6
    :return: dictionary with positions for nodes from networkX graph
    """
    file_url = 'benchmark/layouts/grp_' + str(graph) + '.pkl'
    with open(file_url, 'rb') as fp:
        dict_pos = pickle.load(fp)

    return dict_pos


def get_benchmark_graphs(graph):
    """
    Attention, parameters have to be a string
    :param graph choose between graph 1, 2, 3, 4, 5, 6
    :return: networkX graph
    """
    file_url = 'benchmark/grp_' + str(graph) + '.txt'
    fp = open(file_url, 'r')
    g = generate_graph_from_edgelist(fp)
    fp.close()

    return g

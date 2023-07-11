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
    small_graph, com1 = benchmark.RPG([20, 15, 15], 0.68, 0.32)
    medium_graph, com2 = benchmark.RPG([50, 40, 30, 60, 70], 0.57, 0.23)
    large_graph, com3 = benchmark.RPG([500, 400, 300, 600, 800, 500, 400], 0.76, 0.24)

    grp_small, com4 = benchmark.GRP(50, 13, 5, 0.5, 0.05)
    grp_medium, com5 = benchmark.GRP(250, 45, 10, 0.7, 0.1)
    grp_large, com6 = benchmark.GRP(3500, 463, 7, 0.6, 0.091)

    nx.write_edgelist(small_graph, 'benchmark/rpg_small.txt', delimiter=' ')
    nx.write_edgelist(medium_graph, 'benchmark/rpg_medium.txt', delimiter=' ')
    nx.write_edgelist(large_graph, 'benchmark/rpg_large.txt', delimiter=' ')
    nx.write_edgelist(grp_small, 'benchmark/grp_small.txt', delimiter=' ')
    nx.write_edgelist(grp_medium, 'benchmark/grp_medium.txt', delimiter=' ')
    nx.write_edgelist(grp_large, 'benchmark/grp_large.txt', delimiter=' ')


def create_graph_layouts():
    """
    function was used one time for each graph, no further usage needed
    """
    # load and create graph
    fp1 = open('benchmark/grp_medium.txt')
    g = generate_graph_from_edgelist(fp1)
    fp1.close()

    # choose place to store .pkl file
    with open('benchmark/layouts/grp_medium.pkl', 'wb') as fp2:
        pickle.dump(nx.spring_layout(g), fp2)


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


def get_layout(size, generator_type):
    """
    Attention, parameters have to be a string
    :param size: small, medium, large
    :param generator_type: rpg, grp
    :return: dictionary with positions for nodes from networkX graph
    """
    file_url = 'benchmark/layouts/' + generator_type + '_' + size + '.pkl'
    with open(file_url, 'rb') as fp:
        dict_pos = pickle.load(fp)

    return dict_pos


def get_benchmark_graphs(size, generator_type):
    """
    Attention, parameters have to be a string
    :param size: small, medium, large
    :param generator_type: rpg, grp
    :return: networkX graph
    """
    file_url = 'benchmark/' + generator_type + '_' + size + '.txt'
    fp = open(file_url, 'r')
    g = generate_graph_from_edgelist(fp)
    fp.close()

    return g

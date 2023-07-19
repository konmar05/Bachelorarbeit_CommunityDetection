import pickle
import json
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
    for i in range(1, 7):
        # load and create graph
        fp1 = open('benchmark/grp_' + str(i) + '.txt', 'r')
        g = generate_graph_from_edgelist(fp1)
        fp1.close()

        # choose place to store .pkl file
        with open('benchmark/layouts/grp_' + str(i) + '.pkl', 'wb') as fp2:
            pickle.dump(nx.spring_layout(g), fp2)
        fp2.close()


# todo
def create_bar_diagramm():

    plt.figure(figsize=(15, 7))
    # Paare von Datenwerten
    x_values = []
    y_values = []

    fp = open('evaluation/tmp.txt', 'r')
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
                 'avg_centrality': average_centrality(g)}

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
              'modularity_overlap': evaluation.modularity_overlap(graph, node_clustering_obj),
              'girvan_newman': evaluation.newman_girvan_modularity(graph, node_clustering_obj),
              'z_modularity': evaluation.z_modularity(graph, node_clustering_obj)}

    return scores


def get_scores(graph, node_clustering_obj):
    """

    :param graph: networkX graph
    :param node_clustering_obj: community nodes as NodeClustering object
    :return: a nested dictionary with all modularity and fitness scores calculated from one algorithm
    """
    data = {'modularity_scores': get_modularity_scores(graph, node_clustering_obj),
            'fitness_scores': get_fitness_scores(graph, node_clustering_obj)}

    return data


def get_data(graph=None, algorithm=None, type_of_score=None, score=None):

    with open('evaluation/alternative.json', 'r') as file:
        tmp = json.load(file)

    if graph is None and algorithm is None and type_of_score is None and score is None:
        return tmp
    elif algorithm is None and type_of_score is None and score is None:
        return tmp[graph]
    elif type_of_score is None and score is None:
        return tmp[graph][algorithm]
    elif score is None:
        return tmp[graph][algorithm][type_of_score]
    else:
        return tmp[graph][algorithm][type_of_score][score]

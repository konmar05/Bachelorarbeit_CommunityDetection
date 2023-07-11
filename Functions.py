import networkx as nx
import matplotlib.pyplot as plt
from cdlib import algorithms, viz, evaluation, benchmark, datasets


def generate_graph_from_edgelist(filepointer):

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
    print('Erkannte Communities: ', len(com_result.communities), ' -|- Modularity-Score = ', score.modularity(graph, com_result.communities))
    print('avg_distance: ', evaluation.avg_distance(graph, com_result))
    print('average_internal_degree: ', evaluation.average_internal_degree(graph, com_result))
    print('edges_inside: ', evaluation.edges_inside(graph, com_result))
    print('internal_edge_density: ', evaluation.internal_edge_density(graph, com_result))
    print('size: ', evaluation.size(graph, com_result))
    print('hub_dominance: ', evaluation.hub_dominance(graph, com_result))
    print('\n')


def generate_benchmark_graphs():
    small_graph, com1 = benchmark.RPG([20, 15, 15], 0.68, 0.32)
    medium_graph, com2 = benchmark.RPG([50, 40, 30, 60, 70], 0.57, 0.23)
    large_graph, com3 = benchmark.RPG([500, 400, 300, 600, 800, 500, 400], 0.76, 0.24)

    grp_small, com4 = benchmark.GRP(50, 13, 5, 0.5, 0.05)
    grp_medium, com5 = benchmark.GRP(250, 45, 10, 0.7, 0.1)
    grp_large, com6 = benchmark.GRP(3500, 463, 7, 0.6, 0.091)

    nx.write_edgelist(small_graph, 'benchmark/small_graph.txt', delimiter=' ')
    nx.write_edgelist(medium_graph, 'benchmark/medium_graph.txt', delimiter=' ')
    nx.write_edgelist(large_graph, 'benchmark/large_graph.txt', delimiter=' ')
    nx.write_edgelist(grp_small, 'benchmark/grp_small.txt', delimiter=' ')
    nx.write_edgelist(grp_medium, 'benchmark/grp_medium.txt', delimiter=' ')
    nx.write_edgelist(grp_large, 'benchmark/grp_laarge.txt', delimiter=' ')


def show_benchmark_graphs(size):
    plt.figure(figsize=(10, 10))
    graph = nx.Graph()

    if size == 'small':
        fp = open('benchmark/grp_small.txt', 'r')
        graph = generate_graph_from_edgelist(fp)
    elif size == 'medium':
        fp = open('benchmark/grp_medium.txt', 'r')
        graph = generate_graph_from_edgelist(fp)
    elif size == 'large':
        fp = open('benchmark/grp_laarge.txt', 'r')
        graph = generate_graph_from_edgelist(fp)

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    plt.show()

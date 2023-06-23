"""
@author: Markus Konietzka
@date: 23-06-2023
"""

import networkx as nx
import networkx.algorithms.community.quality as score
import matplotlib.pyplot as plt
from cdlib import algorithms, viz, evaluation, benchmark, datasets


def gen_graph(filepointer):

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


def main():

    fp = open('data/dolphins/out.txt', 'r')
    g = gen_graph(fp)
    pos = nx.spring_layout(g)

    list_results = [algorithms.louvain(g),
                    algorithms.girvan_newman(g, 4),
                    algorithms.label_propagation(g),
                    algorithms.belief(g)]

    for res in list_results:
        viz.plot_network_clusters(g, res, pos, figsize=(10, 10))
        show_fitness_scores(g, res)

    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

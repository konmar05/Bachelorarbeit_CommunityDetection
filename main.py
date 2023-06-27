"""
@author: Markus Konietzka
@date: 23-06-2023
"""

import networkx as nx
import networkx.algorithms.community.quality as score
import matplotlib.pyplot as plt
from cdlib import algorithms, viz, evaluation, benchmark, datasets
from Bundesliga import bundesliga


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

    #fp = open('data/euroroad/out.txt', 'r')
    g = bundesliga('small')
    pos = nx.spring_layout(g)
    com = algorithms.girvan_newman(g, 5)
    lv = algorithms.louvain(g)
    rw = algorithms.walktrap(g)
    #print('gw', score.modularity(g, com.communities))
    #print('lv', score.modularity(g, lv.communities))

    show_fitness_scores(g, com)
    show_fitness_scores(g, lv)
    show_fitness_scores(g, rw)

    #nx.draw(g, pos, with_labels=True)
    viz.plot_network_clusters(g, lv, pos, figsize=(10, 10))
    viz.plot_network_clusters(g, com, pos, figsize=(10, 10))
    viz.plot_network_clusters(g, rw, pos, figsize=(10, 10))
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

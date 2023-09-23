import matplotlib.pyplot as plt

import Functions as f
from cdlib import algorithms, viz
import json


# todo add parameter to specific algorithm (for example walkscan distance)
def test_algorithm(algorithm):
    """
    function to start twenty (20) test runs
    choose from following algorithms

        louvain

        infomap

        label_propagation

        eigenvector

        girvan_newman (1 to 20 iterations)

        greedy_modularity

        walktrap

        async_fluid (try to find 1 to 20 communities)

        walkscan

        graph_entropy

    :param algorithm: choose algorithm, must be a string
    :return:
    """
    global com
    url = 'evaluation/' + algorithm + '.json'
    graphs = {}
    for i in range(1, 7):
        # local variables
        name = 'graph_' + str(i)
        test_runs = {}
        graph = f.get_benchmark_graphs(i)

        # running community detection algorithm (multiple times)
        for k in range(1, 21):

            # look which test is called
            if algorithm == 'louvain':
                com = algorithms.louvain(graph)
            elif algorithm == 'infomap':
                com = algorithms.infomap(graph)
            elif algorithm == 'label_propagation':
                com = algorithms.label_propagation(graph)
            elif algorithm == 'eigenvector':
                com = algorithms.eigenvector(graph)
            elif algorithm == 'girvan_newman':
                com = algorithms.girvan_newman(graph, k)
            elif algorithm == 'greedy_modularity':
                com = algorithms.greedy_modularity(graph)
            elif algorithm == 'walktrap':
                com = algorithms.walktrap(graph)
            elif algorithm == 'async_fluid':
                com = algorithms.async_fluid(graph, k)
            elif algorithm == 'walkscan':
                com = algorithms.walkscan(graph, k)
            elif algorithm == 'graph_entropy':
                com = algorithms.graph_entropy(graph)

            test_runs['test_' + str(k)] = f.get_scores(graph, com)

            plot_size = (8, 8)
            if i > 2:
                plot_size = (10, 10)

            # just for save the graph plots
            # viz.plot_network_clusters(graph, com, f.get_layout(i), figsize=plot_size)
            # plt.savefig('pictures/' + name + algorithm + 'test_' + str(k) + '.png')

        graphs[name] = test_runs

    # write Projektarbeit to json file
    with open(url, 'w') as file:
        json.dump(graphs, file, indent=4)

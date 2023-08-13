"""
@author: Markus Konietzka
@date: 23-06-2023
"""

import networkx as nx
import networkx.algorithms.community.quality as score
import matplotlib.pyplot as plt
from cdlib import algorithms, viz, evaluation, benchmark, datasets
import Functions as f
import Test as test

import json
import pandas as pd


# creating diagramms from test run
def diagramms_for_graphs():

    x_scale_names = []
    y_scale_density = []
    y_scale_asp = []
    y_scale_edges = []
    y_scale_nodes = []
    y_scale_degree = []

    with open('evaluation/graph_stats.json', 'r') as file:
        tmp = json.load(file)

    for graph, stats in tmp.items():
        x_scale_names.append(graph)

    for idx in x_scale_names:
        y_scale_density.append(tmp[idx]['density'])
        y_scale_asp.append(tmp[idx]['avg_shortest_path'])
        y_scale_edges.append(tmp[idx]['edges'])
        y_scale_nodes.append(tmp[idx]['nodes'])
        y_scale_degree.append(tmp[idx]['avg_degree'])

    plt.bar(x_scale_names, y_scale_density)
    plt.ylabel('density')
    plt.show()

    plt.bar(x_scale_names, y_scale_asp)
    plt.ylabel('avg_shortest_path')
    plt.show()

    plt.bar(x_scale_names, y_scale_edges)
    plt.ylabel('edges')
    plt.show()

    plt.bar(x_scale_names, y_scale_nodes)
    plt.ylabel('nodes')
    plt.show()

    fig, ax = plt.subplots()
    ax.stem(x_scale_names, y_scale_degree)
    #plt.bar(x_scale_names, y_scale_degree)
    plt.ylabel('degree')
    plt.show()


# first test run
def firs_test_run():
    graphs = {}
    for i in range(1, 7):

        # local variables
        name = 'graph_' + str(i)
        algorithm = {}
        data = {}
        graph = f.get_benchmark_graphs(i)

        # running community detection algorithm
        louvain = algorithms.louvain(graph)
        label_propagation = algorithms.label_propagation(graph)
        random_walk = algorithms.walktrap(graph)
        eigenvector = algorithms.eigenvector(graph)
        belief = algorithms.belief(graph)
        infomap = algorithms.infomap(graph)

        # add data to dictionary
        algorithm['louvain'] = f.get_scores(graph, louvain)
        algorithm['lable_propagation'] = f.get_scores(graph, label_propagation)
        algorithm['random_walk'] = f.get_scores(graph, random_walk)
        algorithm['eigenvector'] = f.get_scores(graph, eigenvector)
        algorithm['belief'] = f.get_scores(graph, belief)
        algorithm['infomap'] = f.get_scores(graph, infomap)

        graphs[name] = algorithm

    # write data to json file
    with open('evaluation/fitness_scores.json', 'w') as file:
        json.dump(graphs, file, indent=4)


# second test run
def sec_test_run():
    graphs = {}
    for i in range(1, 7):
        # local variables
        name = 'graph_' + str(i)
        algorithm = {}
        tests = {}
        data = {}
        graph = f.get_benchmark_graphs(i)

        # running community detection algorithm
        for k in range(1, 21):
            louvain = algorithms.louvain(graph)
            label_propagation = algorithms.label_propagation(graph)
            girvan_newman = algorithms.girvan_newman(graph, k)
            async_fluid = algorithms.async_fluid(graph, k)
            random_walk = algorithms.walktrap(graph)
            walkscan = algorithms.walkscan(graph)

            # add data to dictionary
            algorithm['louvain'] = f.get_scores(graph, louvain)
            algorithm['lable_propagation'] = f.get_scores(graph, label_propagation)
            algorithm['girvan_newman'] = f.get_scores(graph, girvan_newman)
            algorithm['async_fluid'] = f.get_scores(graph, async_fluid)
            algorithm['random_walk'] = f.get_scores(graph, random_walk)
            algorithm['walkscan'] = f.get_scores(graph, walkscan)

            tests['run_' + str(k)] = algorithm

        graphs[name] = tests

    # write data to json file
    with open('evaluation/second_test.json', 'w') as file:
        json.dump(graphs, file, indent=4)


def third_test_run():
    graphs = {}
    for i in range(1, 7):
        # local variables
        name = 'graph_' + str(i)
        test_runs = {}
        graph = f.get_benchmark_graphs(i)

        # running community detection algorithm (multiple times)
        for k in range(1, 21):
            com = algorithms.louvain(graph)
            test_runs['test_' + str(k)] = f.get_scores(graph, com)

        graphs[name] = test_runs
        #graphs[name + '_obj'] = graph

    # write data to json file
    with open('evaluation/louvain.json', 'w') as file:
        json.dump(graphs, file, indent=4)

# creating plots from results and scores
def plots(graph, type_of_score, specific_score):

    x_scale = []
    y_scale = []

    with open('evaluation/alternative.json', 'r') as file:
        tmp = json.load(file)

    tmp2 = tmp[graph]
    for algo, scores in tmp2.items():
        x_scale.append(algo)

    for idx in x_scale:
        y_scale.append(tmp2[idx][type_of_score][specific_score][2])

    fig, ax = plt.subplots()
    ax.scatter(x_scale, y_scale)
    plt.grid(axis='y')
    plt.ylabel(specific_score)
    plt.show()


def test_algorithms():
    graph = f.get_benchmark_graphs(1)

    c1 = algorithms.louvain(graph)
    c2 = algorithms.infomap(graph)
    c3 = algorithms.label_propagation(graph)
    c4 = algorithms.eigenvector(graph)
    c5 = algorithms.girvan_newman(graph, 5)
    c6 = algorithms.greedy_modularity(graph)
    c7 = algorithms.spectral(graph, 5)
    c8 = algorithms.walktrap(graph)
    c9 = algorithms.async_fluid(graph, 5)
    c10 = algorithms.walkscan(graph, 5)
    c11 = algorithms.graph_entropy(graph)
    list_com = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]

    c1.to_json()

    data = {}

    for com in list_com:
        data[str(com)] = f.get_scores(graph, com)

    with open('evaluation/test.json', 'w') as file:
        json.dump(data, file, indent=4)



# main function/loop to run script
def main_archiv():

    #print(f.get_data('graph_1', 'louvain', 'modularity_scores', 'erdos_renyi'))
    #f.plot_modularity_scores()
    #sec_test_run()
    # f.plot_modularity_over_runs('graph_1', 'girvan_newman')
    third_test_run()
    #test_algorithms()


def main():
    list_algorithm_for_test = ['louvain',
                               'infomap',
                               'label_propagation',
                               'eigenvector',
                               'girvan_newman',
                               'greedy_modularity',
                               #'spectral',
                               'walktrap',
                               'async_fluid',
                               'walkscan',
                               'graph_entropy']

    for algo in list_algorithm_for_test:
        test.test_algorithm(algo)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


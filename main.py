"""
@author: Markus Konietzka
@date: 23-06-2023
"""

import networkx as nx
import networkx.algorithms.community.quality as score
import matplotlib.pyplot as plt
from cdlib import algorithms, viz, evaluation, benchmark, datasets
import Functions as f

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
    with open('evaluation/alternative.json', 'w') as file:
        #fd = f.format_data(graphs)
        #file.write(fd)
        tmp = json.dumps(graphs, indent=4)
        json.dump(graphs, file, indent=4)
        print(tmp)


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




# main function/loop to run script
def main():

    for i in range(1,7):
        g = f.get_benchmark_graphs(i)
        pos = f.get_layout(i)

        nx.draw(g, pos)
        plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #main()
    firs_test_run()
    #plots('graph_1', 'fitness_scores', 'edges_inside')
    #plots('graph_1', 'modularity_scores', 'link')
    #plots('graph_1', 'modularity_scores', 'modularity_density')
    #plots('graph_1', 'modularity_scores', 'modularity_overlap')
    #plots('graph_1', 'modularity_scores', 'girvan_newman')
    #plots('graph_1', 'modularity_scores', 'z_modularity')


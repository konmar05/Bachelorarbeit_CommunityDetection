"""
@author: Markus Konietzka
@date: 23-06-2023
"""

import networkx as nx
import networkx.algorithms.community.quality as score
import matplotlib.pyplot as plt
from cdlib import algorithms, viz, evaluation, benchmark, datasets
import Functions as f



#todo -> test with some algorithm, create function for styling and writing in json file
def first_try_with_several_algortihms():
    data_lp = {}
    data_lou = {}
    data_rw = {}
    data_ev = {}

    scores_lp = {}
    scores_lou = {}
    scores_rw = {}
    scores_ev = {}

    graph = f.get_benchmark_graphs("small", "grp") #todo -> function call incorrect




# prototype with girvan-newmann
def res_girvan_newman():
    length = {}
    modularity = {}
    scores = {}
    data = {}

    graph = f.get_benchmark_graphs("small", "grp")
    for i in range(0, 20):
        attempt = i + 1
        com = algorithms.girvan_newman(graph, attempt)

        length[attempt] = len(com.communities)
        modularity[attempt] = score.modularity(graph, com.communities)
        #todo nested dict for fitness scores

    data = {'length': length, 'modularity': modularity}

    with open('results/graph_stats.json', 'a') as file:
        # json.dump(data, file, indent=None, separators=(", ", ": "), ensure_ascii=False)

        formatted_data = f.format_data(data)
        file.write(formatted_data)


# main function/loop to run script
def main():

    graphs = {}
    for i in range(1, 7):

        name = 'graph_' + str(i)
        algorithm = {}

        graph = f.get_benchmark_graphs(i)

        louvain = algorithms.louvain(graph)
        label_propagation = algorithms.label_propagation(graph)
        random_walk = algorithms.walktrap(graph)
        eigenvector = algorithms.eigenvector(graph)
        belief = algorithms.belief(graph)
        infomap = algorithms.infomap(graph)

        algorithm['louvain'] = f.get_fitness_scores(graph, louvain)
        algorithm['lable_propagation'] = f.get_fitness_scores(graph, label_propagation)
        algorithm['random_walk'] = f.get_fitness_scores(graph, random_walk)
        algorithm['eigenvector'] = f.get_fitness_scores(graph, eigenvector)
        algorithm['belief'] = f.get_fitness_scores(graph, belief)
        algorithm['infomap'] = f.get_fitness_scores(graph, infomap)

        graphs[name] = algorithm

    with open('results/fitness_scores.json', 'w') as file:
        fd = f.format_data(graphs)
        file.write(fd)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

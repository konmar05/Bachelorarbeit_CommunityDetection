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


    label_propagation = algorithms.label_propagation(graph)
    louvain = algorithms.louvain(graph)
    randomwalk = algorithms.walktrap(graph)
    eigenvector = algorithms.eigenvector(graph)


    with open('testdaten/results.json', 'a') as file:
        # json.dump(data, file, indent=None, separators=(", ", ": "), ensure_ascii=False)

        formatted_data = f.format_data(data)
        file.write(formatted_data)


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

    with open('testdaten/results.json', 'a') as file:
        # json.dump(data, file, indent=None, separators=(", ", ": "), ensure_ascii=False)

        formatted_data = f.format_data(data)
        file.write(formatted_data)


# main function/loop to run script
def main():

    for i in range(1, 7):
        g = f.get_benchmark_graphs(i)
        pos = f.get_layout(i)

        nx.draw(g, pos)
        plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

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


def alternative():
    fp = open('benchmark/grp_small.txt')
    g = f.generate_graph_from_edgelist(fp)
    fp.close()

    com = algorithms.label_propagation(g)
    f.show_modularity_scores(g, com)
    print('modularity: ', score.modularity(g, com.communities))
    viz.plot_network_clusters(g, com, position=nx.spring_layout(g))
    plt.show()


# Funktion zum Formatieren der JSON-Daten
def format_json(data, indent=0):
    result = ""
    for key, value in data.items():
        result += " " * indent  # Einrückung entsprechend der Tiefe
        result += f'"{key}": '
        if isinstance(value, dict):
            result += "{\n" + format_json(value, indent + 4) + " " * indent + "}\n"
        else:
            result += f'"{value}"\n'
    return result


#todo -> minor changes to split for every algorithm (results and scores)
def first_try_with_several_algortihms():
    data_lp = {}
    data_lou = {}
    data_rw = {}
    data_ev = {}

    scores_lp = {}
    scores_lou = {}
    scores_rw = {}
    scores_ev = {}

    graph = f.get_benchmark_graphs("small", "grp")

    for i in range(0, 20):
        attempt = i + 1
        label_propagation = algorithms.label_propagation(graph)
        louvain = algorithms.louvain(graph)
        randomwalk = algorithms.walktrap(graph)
        eigenvector = algorithms.eigenvector(graph)

        data_lp[attempt] = len(label_propagation.communities)
        data_lou[attempt] = len(louvain.communities)
        data_rw[attempt] = len(randomwalk.communities)
        data_ev[attempt] = len(eigenvector.communities)

        scores_lp[attempt] = score.modularity(graph, label_propagation.communities)
        scores_lou[attempt] = score.modularity(graph, louvain.communities)
        scores_rw[attempt] = score.modularity(graph, randomwalk.communities)
        scores_ev[attempt] = score.modularity(graph, eigenvector.communities)

    data = {"scores_lp": scores_lp, "scores_lou": scores_lou, "scores_rw": scores_rw, "scores_ev": scores_ev,
            "data_lp": data_lp, "data_lou": data_lou, "data_rw": data_rw, "data_ev": data_ev}

    with open('testdaten/results.json', 'a') as file:
        # json.dump(data, file, indent=None, separators=(", ", ": "), ensure_ascii=False)

        formatted_data = format_json(data)
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

        formatted_data = format_json(data)
        file.write(formatted_data)

# main function/loop to run script
def main():

    res_girvan_newman()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

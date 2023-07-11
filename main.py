"""
@author: Markus Konietzka
@date: 23-06-2023
"""

import networkx as nx
import networkx.algorithms.community.quality as score
import matplotlib.pyplot as plt
from cdlib import algorithms, viz, evaluation, benchmark, datasets
from Bundesliga import bundesliga

import Functions as f


def main():

    fp = open('benchmark/grp_laarge.txt', 'r')
    g = f.generate_graph_from_edgelist(fp)
    pos = nx.spring_layout(g)

    com = algorithms.louvain(g)
    print(score.modularity(g, com.communities))

    #viz.plot_network_clusters(g, com, pos, plot_labels=True)
    #plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

"""
@author: Markus Konietzka
@date: 23-06-2023
"""

import networkx as nx
import networkx.algorithms.community.quality as score
import matplotlib.pyplot as plt
from cdlib import algorithms, viz, evaluation, benchmark, datasets
import Functions as f
import variables as v
import Test as test
import json
import pandas as pd


# creating diagramms for graph stats
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
    plt.ylabel('degree')
    plt.show()


def plot_scores(g):
    """
    Generates subplots for all fitness_scores

    :param g: choose between 'graph_1' to 'graph_6'
    :return:
    """

    fig, axs = plt.subplots(nrows=2, ncols=5, layout='constrained', sharey='row', figsize=(11, 5))
    fig.suptitle('fitness scores')

    y_scale = [v.lv, v.im, v.gn, v.wt, v.gm, v.af]

    x_communities = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_com),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_com),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_com),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_com),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_com),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_com)
    ]

    x_distance = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_dist),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_dist),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_dist),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_dist),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_dist),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_dist)

    ]

    x_embeddedness = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_embd),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_embd),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_embd),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_embd),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_embd),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_embd)
    ]

    x_internal_degree = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_iDeg),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_iDeg),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_iDeg),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_iDeg),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_iDeg),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_iDeg)
    ]

    x_edges_inside = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_edges),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_edges),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_edges),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_edges),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_edges),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_edges)
    ]

    x_expansion = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_exp),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_exp),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_exp),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_exp),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_exp),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_exp)
    ]

    x_internal_edge_density = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_iEDen),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_iEDen),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_iEDen),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_iEDen),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_iEDen),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_iEDen)
    ]

    x_scaled_density = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_scDen),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_scDen),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_scDen),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_scDen),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_scDen),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_scDen)
    ]

    x_transitivity = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_trans),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_trans),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_trans),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_trans),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_trans),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_trans)
    ]

    x_size = [
        f.get_eval_scores(algorithm=y_scale[0], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_size),
        f.get_eval_scores(algorithm=y_scale[1], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_size),
        f.get_eval_scores(algorithm=y_scale[2], graph=g, test='test_3', t_score=v.f_scores, s_score=v.f_size),
        f.get_eval_scores(algorithm=y_scale[3], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_size),
        f.get_eval_scores(algorithm=y_scale[4], graph=g, test='test_1', t_score=v.f_scores, s_score=v.f_size),
        f.get_eval_scores(algorithm=y_scale[5], graph=g, test='test_4', t_score=v.f_scores, s_score=v.f_size)
    ]

    # add supplots
    axs[0, 0].scatter(x_communities, y_scale)
    axs[0, 1].scatter(x_distance, y_scale)
    axs[1, 1].scatter(x_embeddedness, y_scale)
    axs[1, 4].scatter(x_internal_degree, y_scale)
    axs[0, 4].scatter(x_edges_inside, y_scale)
    axs[1, 0].scatter(x_expansion, y_scale)
    axs[0, 2].scatter(x_internal_edge_density, y_scale)
    axs[1, 2].scatter(x_scaled_density, y_scale)
    axs[1, 3].scatter(x_transitivity, y_scale)
    axs[0, 3].scatter(x_size, y_scale)

    # set supplot titles
    axs[0, 0].set_title(v.f_com)
    axs[0, 1].set_title(v.f_dist)
    axs[1, 1].set_title(v.f_embd)
    axs[1, 4].set_title(v.f_iDeg)
    axs[0, 4].set_title(v.f_edges)
    axs[1, 0].set_title(v.f_exp)
    axs[0, 2].set_title(v.f_iEDen)
    axs[1, 2].set_title(v.f_scDen)
    axs[1, 3].set_title(v.f_trans)
    axs[0, 3].set_title(v.f_size)

    plt.show()


def plot_best_results(graph='graph_1', type_of_score='modularity_scores', specific_score='girvan_newman'):
    """

    Generates a plot, where you can see all the results from a fitness or modularity score in every test

    :param graph: choose between 'graph_1' to 'graph_6'
    :param type_of_score: choose between (fitness_scores, modularity_scores) (default is 'modularity_scores')
    :param specific_score: choose a specific score according to his type, more info in variables.py (default is 'girvan_newman')
    :return: no return value
    """

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(graph)

    array_scores_gn = get_eval_scores_as_array(graph, algorithm='girvan_newman')
    array_scores_lv = get_eval_scores_as_array(graph, algorithm='louvain')
    array_scores_lp = get_eval_scores_as_array(graph, algorithm='label_propagation')
    array_scores_ev = get_eval_scores_as_array(graph, algorithm='eigenvector')
    array_scores_im = get_eval_scores_as_array(graph, algorithm='infomap')
    array_scores_rw = get_eval_scores_as_array(graph, algorithm='walktrap')
    array_scores_gm = get_eval_scores_as_array(graph, algorithm='greedy_modularity')
    array_scores_af = get_eval_scores_as_array(graph, algorithm='async_fluid')

    x_scale = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20']
    plt.plot(x_scale, array_scores_lv[1], label='louvain')
    plt.plot(x_scale, array_scores_gn[1], label='girvan_newman')
    plt.plot(x_scale, array_scores_lp[1], label='label_propagation')
    plt.plot(x_scale, array_scores_ev[1], label='eigenvector')
    plt.plot(x_scale, array_scores_im[1], label='infomap')
    plt.plot(x_scale, array_scores_rw[1], label='random_walk')
    plt.plot(x_scale, array_scores_gm[1], label='greedy_modularity')
    plt.plot(x_scale, array_scores_af[1], label='async_fluid')

    plt.grid(axis='y')
    plt.yticks([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    plt.grid(axis='x')
    plt.ylabel('modularity')
    plt.legend()
    plt.show()


# get list with scores according to parameter from function
def get_eval_scores_as_array(graph='graph_1', type_of_score='modularity_scores', specific_score='girvan_newman', algorithm='girvan_newman'):
    """
    return a tuple of arrays

    index 0 is for x_scale

    index 1 is for y_scale
    :param graph: choose between graph_1 to graph_6
    :param type_of_score: choose between "modularity_scores" and "fitness_scores"
    :param specific_score: choose score

            m_scores: (erdos_renyi, link, modularity_density, girvan_newman, z_modularity)

            f_scores: (communities, avg_distance, avg_embeddedness, avg_internal_degree, edges_inside, expansion, internal_edge_density_ scaled_density_ avg_tramstivity, size)

    :param algorithm: choose algorithm (louvain, infomap, label_propagation, eigenvector, girvan_newman , greedy_modularity, walktrap, async_fluid, walkscan, graph_entropy
    :return: returns a tuple of arrayswith the scores according to parameters , index 0 is for x_scale and shows the Test-Nbr, index 1 is for y_scale and holds the values
    """
    file_url = 'evaluation/' + algorithm + '.json'

    with open(file_url, 'r') as file:
        tmp = json.load(file)

    x_scale = []
    y_scale_scores = []
    tests = tmp[graph]
    for test_nbr, t_scores in tests.items():
        x_scale.append(test_nbr)

    for idx in x_scale:
        y_scale_scores.append(tests[idx][type_of_score][specific_score][2])

    return x_scale, y_scale_scores

# creating plots from results and scores
def plots(graph='graph_1', type_of_score='modularity_scores', specific_score='girvan_newman', algorithm1='girvan_newman',
          algorithm2=None, algorithm3=None, algorithm4=None, algorithm5=None, algorithm6=None, algorithm7=None, algorithm8=None):

    file_url = 'evaluation/' + algorithm1 + '.json'
    x_scale = []
    list_alog1 = []
    list_alog2 = []
    list_alog3 = []
    list_alog4 = []
    list_alog5 = []
    list_alog6 = []

    with open(file_url, 'r') as file:
        tmp = json.load(file)

    tests = tmp[graph]
    for test_nbr, t_scores in tests.items():
        x_scale.append(test_nbr)

    for idx in x_scale:
        list_alog1.append(tests[idx][type_of_score][specific_score][2])

    fig = plt.figure(figsize=(20, 10))

    plt.scatter(x_scale, list_alog1)
    plt.grid(axis='y')
    plt.ylabel(specific_score)
    plt.show()


def run_test():
    """
    function to run all test, for each algorithm on every graph
    :return:
    """
    list_algorithm_for_test = ['louvain',
                               'infomap',
                               'label_propagation',
                               'eigenvector',
                               'girvan_newman',
                               'greedy_modularity',
                               'walktrap',
                               'async_fluid',
                               'walkscan',
                               'graph_entropy']

    for algo in list_algorithm_for_test:
        test.test_algorithm(algo)


# main function/loop to run script
def main():

    # print(f.get_data('graph_1', 'louvain', 'modularity_scores', 'erdos_renyi'))
    # f.plot_modularity_scores()
    # sec_test_run()
    # f.plot_modularity_over_runs('graph_1', 'girvan_newman')
    # third_test_run()
    # test_algorithms()
    # f.plot_benchmark_graphs()

    print(f.get_eval_scores(algorithm=v.lv, graph='graph_1', test='test_1', t_score=v.f_scores, s_score=v.f_com))
    plot_scores('graph_1')
    plot_best_results('graph_1')
    plot_best_results('graph_2')
    plot_best_results('graph_3')
    plot_best_results('graph_4')
    plot_best_results('graph_5')
    plot_best_results('graph_6')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main()


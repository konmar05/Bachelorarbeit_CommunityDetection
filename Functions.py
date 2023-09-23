import pickle
import json
import networkx as nx
import networkx.algorithms.community.quality as m
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import Variables as v
import Test as test
from cdlib import algorithms, viz, evaluation, benchmark, datasets


def generate_graph_from_edgelist(filepointer):
    """
    reads an edge list and creates a complete networkX Graph
    file has to be in following form:
    >> 1 3
    >> 3 6
    each line is represents an edge, between to nodes, nodes have to be seperated by a whitespace
    :param filepointer: to read from the given file
    :return: networkX graph
    """
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


def format_data(data, indent=0):
    """
    function to format and style Projektarbeit in a JSON file
    :param data: dictionary to store in JSON file
    :param indent:
    :return: formatted Projektarbeit with indent according to depth
    """
    result = ""
    for key, value in data.items():
        result += " " * indent
        result += f'"{key}": '
        if isinstance(value, dict):
            result += "{\n" + format_data(value, indent + 4) + " " * indent + "}\n"
        else:
            result += f'"{value}"\n'
    return result


def show_fitness_scores(graph, node_clustering_obj):
    """
    functions for having a quick view in cmd-line to the major fitness scores from an algorithm
    :param graph: networkX graph
    :param node_clustering_obj: communities as NodeClustering object
    :return: no return parameter
    """
    print('Erkannte Communities: ', len(node_clustering_obj.communities))
    print('Modularity-Score: ', m.modularity(graph, node_clustering_obj.communities))
    print('avg_distance: ', evaluation.avg_distance(graph, node_clustering_obj))
    print('avg_embeddedness: ', evaluation.avg_embeddedness(graph, node_clustering_obj))
    print('avg_internal_degree: ', evaluation.average_internal_degree(graph, node_clustering_obj))
    print('edges_inside: ', evaluation.edges_inside(graph, node_clustering_obj))
    print('internal_edge_density: ', evaluation.internal_edge_density(graph, node_clustering_obj))
    print('size: ', evaluation.size(graph, node_clustering_obj))
    print('hub_dominance: ', evaluation.hub_dominance(graph, node_clustering_obj))
    print('scaled_density: ', evaluation.scaled_density(graph, node_clustering_obj))
    print('\n')


def show_modularity_scores(graph, node_clustering_obj):
    """
    functions for having q quick vieww in cmd-line to the modularity scores
    :param graph: networkX graph
    :param node_clustering_obj: communities as NodeClustering object
    :return: no return parameter
    """
    print('erdos_renyi: ', evaluation.erdos_renyi_modularity(graph, node_clustering_obj))
    print('link: ', evaluation.link_modularity(graph, node_clustering_obj))
    print('modularity_density: ', evaluation.modularity_density(graph, node_clustering_obj))
    print('modularity_overlap: ', evaluation.modularity_overlap(graph, node_clustering_obj))
    print('newman_girvan: ', evaluation.newman_girvan_modularity(graph, node_clustering_obj))
    print('z_modularity: ', evaluation.z_modularity(graph, node_clustering_obj))


def generate_benchmark_graphs():
    """
    function was used once, no further usage needed

    generates the benchmark graphs and saves the edgelist for each graph in a file
    """
    grp_1, com1 = benchmark.GRP(n=50, s=13, v=13, p_in=0.5, p_out=0.05)
    grp_2, com2 = benchmark.GRP(n=50,  s=7, v=7, p_in=0.5, p_out=0.05)

    grp_3, com3 = benchmark.GRP(n=88, s=13, v=13, p_in=0.4, p_out=0.1)
    grp_4, com4 = benchmark.GRP(n=88, s=5, v=5, p_in=0.4, p_out=0.1)

    grp_5, com5 = benchmark.GRP(n=125, s=13, v=13, p_in=0.6, p_out=0.05)
    grp_6, com6 = benchmark.GRP(n=125, s=5, v=5, p_in=0.6, p_out=0.05)

    nx.write_edgelist(grp_1, 'benchmark/grp_1.txt', delimiter=' ')
    nx.write_edgelist(grp_2, 'benchmark/grp_2.txt', delimiter=' ')
    nx.write_edgelist(grp_3, 'benchmark/grp_3.txt', delimiter=' ')
    nx.write_edgelist(grp_4, 'benchmark/grp_4.txt', delimiter=' ')
    nx.write_edgelist(grp_5, 'benchmark/grp_5.txt', delimiter=' ')
    nx.write_edgelist(grp_6, 'benchmark/grp_6.txt', delimiter=' ')


def create_graph_layouts():
    """
    function was used one time for each graph, no further usage needed

    saves the nodepositions from each benachmark graph
    """
    for i in range(1, 7):
        # load and create graph
        fp1 = open('benchmark/grp_' + str(i) + '.txt', 'r')
        g = generate_graph_from_edgelist(fp1)
        fp1.close()

        # choose place to store .pkl file
        with open('benchmark/layouts/grp_' + str(i) + '.pkl', 'wb') as fp2:
            pickle.dump(nx.spring_layout(g), fp2)
        fp2.close()


def get_layout(graph):
    """
    loads the layout from a graph.

    function is used to have the same position for the nodes in every plot of a graph

    :param graph: integer -> choose between graph 1, 2, 3, 4, 5, 6
    :return: dictionary with positions for nodes from networkX graph
    """
    file_url = 'benchmark/layouts/grp_' + str(graph) + '.pkl'
    with open(file_url, 'rb') as fp:
        dict_pos = pickle.load(fp)

    return dict_pos


def get_benchmark_graphs(graph):
    """
    choose benchmark graph

    :param graph: integer -> choose between graph 1, 2, 3, 4, 5, 6
    :return: networkX graph
    """
    file_url = 'benchmark/grp_' + str(graph) + '.txt'
    fp = open(file_url, 'r')
    g = generate_graph_from_edgelist(fp)
    fp.close()

    return g


def plot_benchmark_graphs():
    """
    plots all benchmark graphs and saves the pictures

    :return:
    """

    for i in range(1, 7):

        plt.figure(figsize=(10, 10))
        graph = get_benchmark_graphs(i)
        pos = get_layout(i)

        nx.draw(graph, pos)
        plt.savefig('pictures/graph_' + str(i) + '.png')


def average_degree(graph):
    """
    function to caluclate the average_degree from all nodes in a given graph

    :param graph:
    :return: value for avg_degree
    """
    tmp = nx.degree(graph)
    mean = 0
    for i in tmp:
        mean += i[1]

    return mean / len(tmp)


def average_centrality(graph):
    """
    function to calculate the average centrality from a given graph

    :param graph:
    :return: value for avg_centrality
    """

    tmp = nx.degree_centrality(graph)
    mean = sum(tmp.values()) / len(tmp)
    return mean


def get_graph_stats():
    """
    generates each benchmark graph and calculates the stats for [nodes, edges, density, shortest_path, degree, centrality]

    :return: dictionary with all stats from all benchmark graphs
    """
    data = {}

    for i in range(1, 7):
        g = get_benchmark_graphs(i)
        stats = {'nodes': nx.number_of_nodes(g),
                 'edges': nx.number_of_edges(g),
                 'density': nx.density(g),
                 'avg_shortest_path': nx.average_shortest_path_length(g),
                 'avg_degree': average_degree(g),
                 'avg_centrality': average_centrality(g),
                 }

        name = 'Graph ' + str(i)
        data[name] = stats

    return data


def get_fitness_scores(graph, node_clustering_obj):
    """

    :param graph: networkX graph
    :param node_clustering_obj: NodeClustering object for communities
    :return: dictionary with all fitness scores
    """
    scores = {'communities': len(node_clustering_obj.communities),
              'avg_distance': evaluation.avg_distance(graph, node_clustering_obj),
              'avg_embeddedness': evaluation.avg_embeddedness(graph, node_clustering_obj),
              'avg_internal_degree': evaluation.average_internal_degree(graph, node_clustering_obj),
              'edges_inside': evaluation.edges_inside(graph, node_clustering_obj),
              'expansion': evaluation.expansion(graph, node_clustering_obj),
              'internal_edge_density': evaluation.internal_edge_density(graph, node_clustering_obj),
              'scaled_density': evaluation.scaled_density(graph, node_clustering_obj),
              'avg_transitivity': evaluation.avg_transitivity(graph, node_clustering_obj),
              'size': evaluation.size(graph, node_clustering_obj)}

    return scores


def get_modularity_scores(graph, node_clustering_obj):
    """

    :param graph: networkX graph
    :param node_clustering_obj: communities as NodeClustering object
    :return: dictionary with all modularity scores
    """
    scores = {'erdos_renyi': evaluation.erdos_renyi_modularity(graph, node_clustering_obj),
              'link': evaluation.link_modularity(graph, node_clustering_obj),
              'modularity_density': evaluation.modularity_density(graph, node_clustering_obj),
              'girvan_newman': evaluation.newman_girvan_modularity(graph, node_clustering_obj),
              'z_modularity': evaluation.z_modularity(graph, node_clustering_obj)}

    return scores


def get_scores(graph, node_clustering_obj):
    """

    :param graph: networkX graph
    :param node_clustering_obj: community nodes as NodeClustering object
    :return: a nested dictionary with all modularity and fitness scores calculated from one algorithm
    """
    data = {'communities': node_clustering_obj.communities,
            'modularity_scores': get_modularity_scores(graph, node_clustering_obj),
            'fitness_scores': get_fitness_scores(graph, node_clustering_obj)}

    return data


def get_eval_scores(algorithm, graph=None, test=None, t_score=None, s_score=None):

    with open('evaluation/' + algorithm + '.json', 'r') as file:
        tmp = json.load(file)

    if s_score == 'communities':
        return tmp[graph][test][t_score][s_score]
    else:
        if graph is None and test is None and t_score is None and s_score is None:
            return tmp
        elif test is None and t_score is None and s_score is None:
            return tmp[graph]
        elif t_score is None and s_score is None:
            return tmp[graph][test]
        elif s_score is None:
            return tmp[graph][test][t_score]
        else:
            return tmp[graph][test][t_score][s_score][2]



def get_data(file_url, graph=None, run=None, algorithm=None, type_of_score=None, score=None):

    with open(file_url, 'r') as file:
        tmp = json.load(file)

    if graph is None and run is None and algorithm is None and type_of_score is None and score is None:
        return tmp
    elif run is None and algorithm is None and type_of_score is None and score is None:
        return tmp[graph]
    elif algorithm is None and type_of_score is None and score is None:
        return tmp[graph][run]
    elif type_of_score is None and score is None:
        return tmp[graph][run][algorithm]
    elif score is None:
        return tmp[graph][run][algorithm][type_of_score]
    else:
        return tmp[graph][run][algorithm][type_of_score][score][2]


def save_graph_stats():

    tmp = get_graph_stats()

    with open('evaluation/graph_stats.json', 'w') as file:
        json.dump(tmp, file,  indent=4)


def plot_graph_stats():

    with open('evaluation/graph_stats.json', 'r') as file:
        tmp = json.load(file)

    x_axes = ['Graph 1', 'Graph 2', 'Graph 3', 'Graph 4', 'Graph 5', 'Graph 6']
    y_nodes = []
    y_edges = []
    y_density = []
    y_shortest_path = []
    y_degree = []

    for g in tmp:
        a = tmp[g]
        y_nodes.append(a['nodes'])
        y_edges.append(a['edges'])
        y_density.append(a['density'])
        y_shortest_path.append(a['avg_shortest_path'])
        y_degree.append(a['avg_degree'])

    plt.figure(figsize=(20, 12))
    grid = gridspec.GridSpec(3, 3)

    # local var for styling plots
    ls = ':'
    lw = 1

    axes_1 = plt.subplot(grid[0, :])
    axes_1.set_title('nodes')
    axes_1.scatter(x_axes, y_nodes)
    axes_1.axhline(min(y_nodes), c='green', ls=ls, lw=lw)
    axes_1.axhline((min(y_nodes) + max(y_nodes))/2, c='orange', ls=ls, lw=lw)
    axes_1.axhline(max(y_nodes), c='red', ls=ls, lw=lw)
    plt.ylim([min(y_nodes)-10, max(y_nodes)+10])

    axes_2 = plt.subplot(grid[1, :-1])
    axes_2.set_title('edges')
    axes_2.scatter(x_axes, y_edges)
    for i in y_edges:
        axes_2.axhline(i, c='blue', ls=ls, lw=lw)
    plt.ylim([min(y_edges)-50, max(y_edges)+50])

    axes_3 = plt.subplot(grid[1:, -1])
    axes_3.set_title('density')
    axes_3.scatter(x_axes, y_density)
    for i in y_density:
        axes_3.axhline(i, c='blue', ls=ls, lw=lw)
    plt.ylim([min(y_density)-0.01, max(y_density)+0.01])

    axes_4 = plt.subplot(grid[-1, 0])
    axes_4.scatter(x_axes, y_degree)
    axes_4.set_title('degree')
    for i in y_degree:
        axes_4.axhline(i, c='blue', ls=ls, lw=lw)
    plt.ylim([min(y_degree)-1, max(y_degree)+1])

    axes_5 = plt.subplot(grid[-1, -2])
    axes_5.scatter(x_axes, y_shortest_path)
    axes_5.set_title('shortest_path')
    for i in y_shortest_path:
        axes_5.axhline(i, c='blue', ls=ls, lw=lw)
    plt.ylim([min(y_shortest_path)-0.1, max(y_shortest_path)+0.1])

    plt.tight_layout()
    plt.show()


def plot_modularity_scores():

    with open('evaluation/fitness_scores.json', 'r') as file:
        tmp = json.load(file)

    x_axes = ['erdos_renyi', 'link', 'modularity_density', 'modularity_overlap', 'girvan_newman', 'z_modularity']
    l_algorithms = ['louvain', 'lable_propagation', 'random_walk', 'eigenvector', 'belief', 'infomap']

    y_erdos = []
    y_link = []
    y_m_density = []
    y_m_overlap = []
    y_gir_new = []
    y_z_mod = []

    for algo in l_algorithms:
        y_erdos.append(get_data('graph_1', algo, 'modularity_scores', 'erdos_renyi'))

    for algo in l_algorithms:
        y_link.append(get_data('graph_1', algo, 'modularity_scores', 'link'))

    for algo in l_algorithms:
        y_m_density.append(get_data('graph_1', algo, 'modularity_scores', 'modularity_density'))

    for algo in l_algorithms:
        y_m_overlap.append(get_data('graph_1', algo, 'modularity_scores', 'modularity_overlap'))

    for algo in l_algorithms:
        y_gir_new.append(get_data('graph_1', algo, 'modularity_scores', 'girvan_newman'))

    for algo in l_algorithms:
        y_z_mod.append(get_data('graph_1', algo, 'modularity_scores', 'z_modularity'))

    plt.figure(figsize=(20, 12))

    plt.scatter(x_axes, y_erdos)
    plt.scatter(x_axes, y_link)
    plt.scatter(x_axes, y_m_density)
    plt.scatter(x_axes, y_m_overlap)
    plt.scatter(x_axes, y_gir_new)
    plt.scatter(x_axes, y_z_mod)
    plt.yscale('log')

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


def choose_scores(array_algorithm, graph, array_test, t_score, s_score):
    """
    choose a specific score to load from the algorithm.json files

    function is optimzed for use in "plot_scores()"

    :param array_algorithm: array to choose algorithms
    :param graph: choose graph
    :param array_test: array to choose test
    :param t_score: choose type of score
    :param s_score: choose specific score
    :return: returns a array with the specific scores, using as x_scale
    """
    x_scale_array = []

    for i in range(0, len(array_algorithm)):
        tmp = get_eval_scores(
            algorithm=array_algorithm[i],
            graph=graph,
            test='test_'+str(array_test[i]),
            t_score=t_score,

            s_score=s_score)

        x_scale_array.append(tmp)
    return x_scale_array


def plot_scores(g):
    """
    Generates subplots for all fitness_scores

    some arrys need to changed before running the script

    y_scale[] -> array to choose the algorithms

    test_array[] -> array to choose the test for the algorithm (first index refers to first index in y_scale, second to second and so on ...)



    :param g: choose between 'graph_1' to 'graph_6'
    :return:
    """

    fig, axs = plt.subplots(nrows=2, ncols=5, layout='constrained', sharey='row', figsize=(11, 5))
    fig.suptitle('fitness scores')

    y_scale = [v.lv, v.gm, v.af, v.wt, v.ev, v.gn]  # change this  array to choose the algorithms
    test_array = [12, 9, 8, 9, 9, 20]  # change this array to choose the test (1-20)

    x_communities = choose_scores(y_scale, g, test_array, v.f_scores, v.f_com)
    x_distance = choose_scores(y_scale, g, test_array, v.f_scores, v.f_dist)
    x_embeddedness = choose_scores(y_scale, g, test_array, v.f_scores, v.f_embd)
    x_internal_degree = choose_scores(y_scale, g, test_array, v.f_scores, v.f_iDeg)
    x_edges_inside = choose_scores(y_scale, g, test_array, v.f_scores, v.f_edges)
    x_expansion = choose_scores(y_scale, g, test_array, v.f_scores, v.f_exp)
    x_internal_edge_density = choose_scores(y_scale, g, test_array, v.f_scores, v.f_iEDen)
    x_scaled_density = choose_scores(y_scale, g, test_array, v.f_scores, v.f_scDen)
    x_transitivity = choose_scores(y_scale, g, test_array, v.f_scores, v.f_trans)
    x_size = choose_scores(y_scale, g, test_array, v.f_scores, v.f_size)

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
    :param specific_score: choose a specific score according to his type, more info in Variables.py (default is 'girvan_newman')
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
    """
    older function, not used anymore

    :param graph:
    :param type_of_score:
    :param specific_score:
    :param algorithm1:
    :param algorithm2:
    :param algorithm3:
    :param algorithm4:
    :param algorithm5:
    :param algorithm6:
    :param algorithm7:
    :param algorithm8:
    :return:
    """

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

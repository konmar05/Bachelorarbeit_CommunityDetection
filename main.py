"""
@author: Markus Konietzka
@date: 23-06-2023
"""

import Functions as f


# main function/loop to run script
def main():

    f.run_test()

    f.plot_best_results('graph_1')
    f.plot_best_results('graph_2')
    f.plot_best_results('graph_3')
    f.plot_best_results('graph_4')
    f.plot_best_results('graph_5')
    f.plot_best_results('graph_6')

    #f.plot_scores('graph_6')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main()


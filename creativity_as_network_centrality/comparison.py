import argparse
import configure as conf
import pandas as pd
from network_utils import *
from helpers import *
import time
import numpy as np


def random_network_properties(num_networks):
    print("================ Generating random networks ================")
    rand_aspl = []
    rand_cc = []
    rand_mod = []
    for i in range(df.shape[0]):
        print("================================ song {num} / {total} =================================".format(num=i+1, total=df.shape[0]))
        print("================================ song: {name} ================================".format(name=df['songID'][i]))
        g = generate_network(dataInst, df['songID'][i], directed=True)
        g_ = generate_network(dataInst, df['songID'][i], directed=False)
        raspls = []  # random average shortest path lengths
        rccs = []  # random clustering coefficients
        rms = []  # random modularities
        for j in range(num_networks):
            #print(j, '/', num_networks)
            source_node_list, target_node_list, _ = directed_random_net_by_switching_algorithm(g)
            r = two_node_list_to_directed_graph(source_node_list, target_node_list)
            #r = generate_directed_random_networks_by_switching_algorithm(g)
            r_aspl = get_aspl(r)
            raspls.append(r_aspl)

            small_node_list, big_node_list, _ = undirected_random_net_by_switching_algorithm(g_)
            r = two_node_list_to_undirected_graph(small_node_list, big_node_list)
            #r = generate_undirected_random_networks_by_switching_algorithm(g_)

            r_cc = get_cc(r)
            rccs.append(r_cc)
            r_mod = get_random_graph_modularity(r)
            rms.append(r_mod)
        rand_aspl.append(np.mean(raspls))
        rand_cc.append(np.mean(rccs))
        rand_mod.append(np.mean(rms))
    df['rand_cc'] = rand_cc
    df['rand_aspl'] = rand_aspl
    df['rand_mod'] = rand_mod
    return df


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='data analysis')
    parser.add_argument('--number-of-random-networks', type=int, default=conf.number_of_random_networks,
                        help='number of random networks generated for comparison with real melodic networks')
    args = parser.parse_args()

    df = pd.read_excel(os.path.join(conf.result_dir, conf.analysis_filename), index_col=0)
    dataInst = load_data()

    # Add columns of random clustering coefficient and average shortest path length of random networks by switching algorithm.
    start = time.time()
    df_ = random_network_properties(args.number_of_random_networks)
    end = time.time()
    print("generate random networks in {sec} seconds".format(sec=end-start))

    with pd.ExcelWriter(os.path.join(conf.result_dir, conf.analysis_filename)) as writer:
        df_.to_excel(writer)
    print("================ dataframe file saved ================")


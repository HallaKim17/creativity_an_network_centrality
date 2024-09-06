from data_generator import Preprocessing
from data_generator_update import Preprocessing_update
import collections
import numpy as np
import math
import networkx as nx
import argparse
import configure as conf
from helpers import *
import dill
from read_data import ReadDataInfo
from helpers import load_data


def melody_bigram_plane(midi):
    assert isinstance(midi, Preprocessing)
    bigram = midi.bigram()
    bigram = dict(collections.Counter(bigram))
    plane = np.zeros((128,128))
    for (b1, b2) in bigram:
        plane[b1, b2] = bigram[(b1,b2)] / sum(bigram.values())
    return plane

def rhythm_bigram_plane(midi, unique_rhythms):
    assert isinstance(midi, Preprocessing_update)
    bigram = midi.rhythm_bigram()
    bigram = dict(collections.Counter(bigram))
    plane = np.zeros((len(unique_rhythms),len(unique_rhythms)))
    for (b1, b2) in bigram:
        plane[unique_rhythms.index(b1), unique_rhythms.index(b2)] = bigram[(b1,b2)] / sum(bigram.values())
    return plane


def bigram_distance(midi1, midi2, unique_rhythms):
    if len(unique_rhythms) != 0:
        midi1_prob_dist = rhythm_bigram_plane(midi1, unique_rhythms)
        midi2_prob_dist = rhythm_bigram_plane(midi2, unique_rhythms)

        midi1_bigram = dict(collections.Counter(midi1.rhythm_bigram()))
        midi2_bigram = dict(collections.Counter(midi2.rhythm_bigram()))
        all_bigrams = list(set(midi1_bigram) | set(midi2_bigram))

        error_sum = 0
        for (b1, b2) in all_bigrams:
            error = (midi1_prob_dist[unique_rhythms.index(b1), unique_rhythms.index(b2)] - midi2_prob_dist[unique_rhythms.index(b1), unique_rhythms.index(b2)]) ** 2
            error_sum += error
        distance = math.sqrt(error_sum)
        return distance

    else:
        midi1_prob_dist = melody_bigram_plane(midi1)
        midi2_prob_dist = melody_bigram_plane(midi2)

        midi1_bigram = dict(collections.Counter(midi1.bigram()))
        midi2_bigram = dict(collections.Counter(midi2.bigram()))
        all_bigrams = list(set(midi1_bigram) | set(midi2_bigram))

        error_sum = 0
        for (b1, b2) in all_bigrams:
            error = (midi1_prob_dist[b1, b2] - midi2_prob_dist[b1, b2]) ** 2
            error_sum += error
        distance = math.sqrt(error_sum)
        return distance


# calculate similarity between pairs of songs
def similarity(Songs, dataInst, dataInfo, unique_rhythms):
    print("============= calculating similarity between songs =============")

    weighted_edge = []
    n = 0
    total = len(Songs) * (len(Songs)-1) / 2
    for i in range(len(Songs)):
        for j in range(len(Songs)):
            if i < j:
                song1 = Songs[i]
                song2 = Songs[j]
                year1 = float(dataInfo[dataInfo['songID']==song1]['year'])
                year2 = float(dataInfo[dataInfo['songID']==song2]['year'])
                if year1 != year2:
                    distance = bigram_distance(dataInst[song1], dataInst[song2], unique_rhythms)
                    similarity = 1 / (1+distance)
                    if year1 < year2:
                        edge_weight = (song1, song2, similarity)
                        weighted_edge.append(edge_weight)
                    else:
                        edge_weight = (song2, song1, similarity)
                        weighted_edge.append(edge_weight)
            n += 1
            progressBar(n, total)
    return weighted_edge


def creativity_evaluation_network(dataInfo, L, weighted_edge, balancing_percentile=50):
    print("============= constructing a network of songs =============")
    # Set number of incoming edge (L)
    # To have equal amount of penalty, influence in this case, when evaluating creativity score
    topK_weighted_edge = []
    for song in dataInfo['songID']:
        incoming_edge_of_this_song = [(s1,s2,sim) for (s1,s2,sim) in weighted_edge if s2 == song]
        topK = sorted(incoming_edge_of_this_song, key=lambda edge: edge[2], reverse=True)[:L]
        topK_weighted_edge.extend(topK)

    # Sort songs by year of composition
    year_sorted_dataInfo = dataInfo.sort_values(by=['year'])
    year_sorted_dataInfo.head()

    # Balancing function
    # Set percentile of incoming edges to be reversed, which becomes the level of novelty of the node
    balanced_weighted_edge = []
    for song in year_sorted_dataInfo['songID']:
        incoming_edge_of_this_song = [(s1, s2, sim) for (s1,s2,sim) in topK_weighted_edge if s2 == song]
        if len(incoming_edge_of_this_song) == 0:
            continue
        p_percentile = np.percentile([sim for (s1,s2,sim) in incoming_edge_of_this_song], balancing_percentile, interpolation='linear')
        balanced_edge_of_this_song = [(s1,s2,sim-p_percentile) for (s1,s2,sim) in incoming_edge_of_this_song]
        direction_adjusted = [(s2,s1,-sim) if sim<0 else (s1,s2,sim) for (s1,s2,sim) in balanced_edge_of_this_song]
        balanced_weighted_edge.extend(direction_adjusted)

    # Creativity score of each node is divided by the sum of incoming edge weights (received influence).
    W_hat = []
    for song in dataInfo['songID']:
        incoming_edge_of_this_song = [(s1,s2,sim) for (s1,s2,sim) in balanced_weighted_edge if s2 == song]
        incoming_sum = sum([sim for (s1,s2,sim) in incoming_edge_of_this_song])
        if incoming_sum == 0:
            continue
        normalized_edge = [(s1,s2,sim/incoming_sum) for (s1,s2,sim) in incoming_edge_of_this_song]
        W_hat.extend(normalized_edge)

    # generate a network whose adjacency matrix contains novelty and influence of each node
    G = nx.DiGraph()
    for (s1,s2,normalized_edge) in W_hat:
        G.add_edge(s1,s2, weight=normalized_edge)

    return G


def inverted_pagerank_creativity(G, initialize=None, max_iter=100000, converge_error=0.01):
    print("============= computing creativity =============")
    """
    Arguments
    _________
    G: networkx Digraph.
    initialize: initializing value for all nodes, float
    """
    nodes = list(G.nodes)
    score = np.array([initialize] * len(nodes))
    assert score.shape[0] == len(nodes)

    adjacency_matrix = nx.adjacency_matrix(G, nodelist=nodes, weight='weight').todense()

    for i in range(max_iter):
        score_ = np.dot(adjacency_matrix, score)
        if np.linalg.norm(score_ - score) < converge_error:
            print('converge_error: ', np.linalg.norm(score_ - score))
            break
        if score_.shape[0] != len(nodes):
            score = np.transpose(score_)
    print('converge_error: ', np.linalg.norm(score_ - score))
    creativities = score_ / np.linalg.norm(score_)
    creativities = creativities.tolist()
    creativity = {node: creativity[0] for (node, creativity) in zip(nodes, creativities)}
    return creativity


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='creativity computation')
    parser.add_argument('--element', type=str, required=True, help='which element to analyze data, rhythm or melody')
    parser.add_argument('--L-min', type=int, default=conf.L_min, help='minimum number of incoming edges')
    parser.add_argument('--L-max', type=int, default=conf.L_max, help='maximum number of incoming edges')
    parser.add_argument('--L-step', type=int, default=conf.L_step, help='spacing between L values')
    parser.add_argument('--max-iter', type=int, default=conf.max_iter, help='number of iterations')
    parser.add_argument('--converge-error', type=int, default=conf.converge_error, help='convergence error for computing creativity')
    parser.add_argument('--balancing-percentile', type=int, default=conf.balancing_percentile,
                        help='percentile of the incoming weights below which is considered as novelty')
    args = parser.parse_args()

    dataInst = load_data(element=args.element)
    dataInfo = ReadDataInfo(dataInfo_dir=conf.dataInfo_dir)
    songs = list(dataInst.keys())

    weighted_edge_file_name = None
    creativity_scores_file_name = None
    if args.element == 'rhythm':
        weighted_edge_file_name = conf.similarity_rhythm_filename
        creativity_scores_file_name = conf.creativity_score_rhythm_filename
    elif args.element == 'melody':
        weighted_edge_file_name = conf.similarity_filename
        creativity_scores_file_name = conf.creativity_score_filename

    # get all unique rhythms in the whole dataset
    unique_rhythms = []
    if args.element == 'rhythm':
        for song in dataInst:
            rhy = list(set(dataInst[song].rhythm()))
            unique_rhythms.extend(rhy)
    unique_rhythms = sorted(list(set(unique_rhythms)))
    print("Num of unique rhythms in dataset: {}".format(len(unique_rhythms)))

    # computing similarity
    weighted_edge = similarity(songs, dataInst, dataInfo, unique_rhythms)
    with open(os.path.join(conf.result_dir, weighted_edge_file_name), "wb") as f:
        dill.dump(weighted_edge, f)

    # computing creativity
    creativity_scores = {song: [] for song in songs}
    for L in np.arange(args.L_min, args.L_max + 1, args.L_step):
        print("L = %s" % L)
        G = creativity_evaluation_network(dataInfo, L, weighted_edge, args.balancing_percentile)
        creativity_given_L = inverted_pagerank_creativity(G, initialize=1 / len(songs), max_iter=args.max_iter,
                                                          converge_error=args.converge_error)
        for song in creativity_given_L:
            creativity_scores[song].append(creativity_given_L[song])
    mean_creativity_scores = {song: np.mean(creativity_scores[song]) for song in creativity_scores}
    with open(os.path.join(conf.result_dir, creativity_scores_file_name), "wb") as f:
        dill.dump(mean_creativity_scores, f)

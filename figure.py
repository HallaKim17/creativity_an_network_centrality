import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import collections
import numpy as np
from scipy.optimize import curve_fit
import copy
import math
import pandas as pd
from network_utils import *
from scipy.stats import pearsonr
from helpers import *
import argparse
import configure as conf
import os
from configure import *
import seaborn as sns


def pdf_of_bigrams(df, dataInst, element=None, save_dir=None):
    all_bigrams = []
    for song in df['songID']:
        if element=='melody':
            all_bigrams.extend(dataInst[song].bigram())
        elif element=='rhythm':
            all_bigrams.extend(dataInst[song].rhythm_bigram())
    all_bigrams = dict(collections.Counter(all_bigrams))
    all_bigrams_count = list(all_bigrams.values())
    all_bigrams_unique = list(set(all_bigrams_count))
    all_bigrams_prob = np.array([all_bigrams_count.count(n)/len(all_bigrams_count) for n in all_bigrams_unique])

    fig, ax = plt.subplots()
    plt.xlabel('k', fontsize=21, fontstyle='italic')
    plt.loglog(all_bigrams_unique, all_bigrams_prob, color='b', marker='o', linestyle='None', markeredgecolor='black')
    plt.ylabel(r'$\pi(k)$', fontsize=21, fontstyle='italic')
    plt.grid(linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(os.path.join(save_dir, '[Figure 1] PDF of bigrams'))
    plt.close()
    return all_bigrams_unique, all_bigrams_prob


def histogram_of_edge_weight(weighted_edge, save_dir=None):
    edge_weights = [w[2] for w in weighted_edge]
    fig, ax = plt.subplots()
    plt.xlabel('w', fontsize=21, fontstyle='italic')
    hist, bins = np.histogram(edge_weights, bins=30)
    plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='green', edgecolor='black', linewidth=1.2)
    plt.ylabel('Normalized frequency', fontsize=21, fontstyle='italic')
    plt.savefig(os.path.join(save_dir, '[Figure 2] histogram of edge weight'))
    plt.close()


def edge_weight_heatmap(df, songs, weighted_edge, save_dir=None):
    fig, ax = plt.subplots(figsize=(9, 8))

    corr = pd.DataFrame(data=np.zeros((len(songs),len(songs))), index=songs, columns=songs)
    for w in weighted_edge:
        corr.loc[w[0]][w[1]] = w[2]
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap,
                    xticklabels=[df[df['songID']==s]['year'][0] for s in songs],
                    yticklabels=[df[df['songID']==s]['year'][0] for s in songs])
    g.set_xticklabels(g.get_xticklabels(), rotation=90, fontsize=2)
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=2)
    plt.savefig(os.path.join(save_dir, '[Figure 0] heatmap of edge weight'))
    plt.close()

def song_representation(songID, save_dir=None):
    def melody_bigram_plane(midi):
        bigram = midi.bigram()
        bigram = dict(collections.Counter(bigram))
        plane = np.zeros((128, 128))
        for (b1, b2) in bigram:
            plane[b1, b2] = bigram[(b1, b2)] / sum(bigram.values())
        return plane

    melody_plane = melody_bigram_plane(dataInst[songID])

    melody_plane = pd.DataFrame(melody_plane, index=list(range(0, 128)), columns=list(range(0, 128)))
    fig, ax = plt.subplots(figsize=(9, 8))
    plt.pcolor(melody_plane, cmap='Greys')
    plt.ylabel('First note', fontsize=18, fontstyle='italic')
    plt.xlabel('Second note', fontsize=18, fontstyle='italic')
    # set note ticks at [21,24,36,48,60,72,84,96,108]
    # plt.xticks([0.5,3.5,15.5,27.5,39.5,51.5,63.5,75.5,87.5], ['A0','C1','C2','C3','C4','C5','C6','C7','C8'], rotation=30, fontsize=9)
    # plt.yticks([0.5,3.5,15.5,27.5,39.5,51.5,63.5,75.5,87.5], ['A0','C1','C2','C3','C4','C5','C6','C7','C8'], rotation=30, fontsize=9)
    plt.xticks([0.5, 12.5, 24.5, 36.5, 48.5, 60.5, 72.5, 84.5, 96.5, 108.5, 120.5],
               ['C-1', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'], rotation=30, fontsize=12)
    plt.yticks([0.5, 12.5, 24.5, 36.5, 48.5, 60.5, 72.5, 84.5, 96.5, 108.5, 120.5],
               ['C-1', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'], rotation=30, fontsize=12)
    plt.title('Song representation of %s' % songID)
    plt.colorbar()
    plt.grid(linestyle='--', which='both')
    plt.savefig(os.path.join(save_dir, '[Figure 3] Song representation %s' % songID))
    plt.close()


def song_representation_enlarge(songID, save_dir=None):

    def melody_bigram_plane(midi):
        bigram = midi.bigram()
        bigram = dict(collections.Counter(bigram))
        assert songID == 'Bach103'
        plane = np.zeros((49,49))
        for (b1, b2) in bigram:
            plane[b1-36, b2-36] = bigram[(b1, b2)] / sum(bigram.values())
        return plane

    melody_plane = melody_bigram_plane(dataInst[songID])

    assert songID == 'Bach103'
    melody_plane = pd.DataFrame(melody_plane, index=list(range(0, 49)), columns=list(range(0, 49)))
    fig, ax = plt.subplots(figsize=(9, 8))
    plt.pcolor(melody_plane, cmap='Blues')
    plt.xticks([0.5, 12.5, 24.5, 28.5, 36.5, 48.5],
               ['C2', 'C3', 'C4', 'E4', 'C5', 'C6'], rotation=30, fontsize=12)
    plt.yticks([0.5, 12.5, 21.5, 24.5, 36.5, 48.5],
               ['C2', 'C3', 'A3', 'C4', 'C5', 'C6'], rotation=30, fontsize=12)
    plt.colorbar()
    ticks = [i for i in np.arange(0.5,49.5,1)]
    ax.set_xticks(ticks, minor=True)
    ax.set_yticks(ticks, minor=True)
    plt.grid(linestyle='--', which='both')
    plt.savefig(os.path.join(save_dir, '[Figure 3(Large)] Large song representation %s' % songID))
    plt.close()


def creativity_plot(df, save_dir=None):
    fig, ax = plt.subplots()
    plt.scatter(df['year'],df['centrality'], marker='o', facecolors='hotpink', edgecolors='black', s=9.)
    plt.xlabel('Year', fontsize=15, fontstyle='italic')
    plt.ylabel('Creativity', fontsize=15, fontstyle='italic')
    plt.axvline(conf.Baroque_end, linestyle='--', linewidth=0.8, color='blueviolet')
    plt.axvline(conf.Classical_end, linestyle='--', linewidth=0.8, color='blueviolet')
    plt.axvline(conf.Romantic_start, linestyle='--', linewidth=0.8, color='blueviolet')
    plt.axvline(conf.Romantic_end, linestyle='--', linewidth=0.8, color='blueviolet')
    plt.axvline(conf.Post_romantic_end, linestyle='--', linewidth=0.8, color='blueviolet')
    plt.axvline(conf.Modern_start, linestyle='--', linewidth=0.8, color='blueviolet')
    plt.axvline(conf.Modern_end, linestyle='--', linewidth=0.8, color='blueviolet')
    plt.axvline(conf.Twentieth_century_start, linestyle='--', linewidth=0.8, color='blueviolet')
    plt.savefig(os.path.join(save_dir, '[Figure 4] creativity by year'))
    plt.close()


def creative_network(dataInst, songID, save_dir=None):
    from pyvis.network import Network
    fig, ax = plt.subplots()
    net = Network()
    nodes = list(set(dataInst[songID].melody()))
    edges = dataInst[songID].bigram()
    weight = dict(collections.Counter(edges))
    net.add_nodes(nodes,
                  label=nodes)
    for key in weight:
        net.add_edge(key[0],key[1], value=weight[key])

    #net.show('network.html')
    plt.savefig(os.path.join(save_dir, '[Figure 0] creative network'))
    plt.close()


def creativity_and_number_of_new_nodes(df, save_dir=None):
    fig, ax = plt.subplots()
    plt.scatter(df['centrality'], df['new_nodes'], marker='o', facecolors='hotpink', edgecolors='black', s=9.)
    plt.xscale('log')
    plt.xlabel(r'$log_{10}{creativity}$', fontsize=15)
    plt.xlabel('Creativity', fontsize=15)
    plt.ylabel('Number of new nodes', fontsize=15)
    plt.savefig(os.path.join(save_dir, '[Figure 5] creativity and number of new nodes'))
    plt.close()


def creativity_and_number_of_new_nodes_update(df, save_dir=None):

    g = sns.JointGrid(data=df, x="centrality", y="new_nodes", marginal_ticks=True)

    # Set a log scaling on the y axis
    g.ax_joint.set(xscale="log")

    g.plot_joint(sns.scatterplot, s=70, alpha=.6, color='#FC7F77', marker='p')

    sns.histplot(x=df['centrality'], fill=True, linewidth=1, ax=g.ax_marg_x, kde=True, color='#FC7F77')
    sns.histplot(y=df['new_nodes'], fill=True, linewidth=1, ax=g.ax_marg_y, binwidth=1, discrete=True,
                 kde=True, line_kws={'color':'k', 'linewidth':1},
                 color='#FC7F77')
    g.ax_joint.set_xlabel('Creativity', fontsize=15)
    g.ax_joint.set_ylabel('Number of new nodes', fontsize=15)

    plt.savefig(os.path.join(save_dir, '[Figure 5_update] creativity and number of new nodes'))
    plt.close()

def creativity_and_number_of_new_edges(df, save_dir=None):
    fig, ax = plt.subplots()
    plt.scatter(df['centrality'], df['new_edges'], marker='o', facecolors='hotpink', edgecolors='black', s=9.)
    plt.xscale('log')
    plt.xlabel(r'$log_{10}{creativity}$', fontsize=15)
    plt.xlabel('Creativity', fontsize=15)
    plt.ylabel('Number of new edges', fontsize=15)
    plt.savefig(os.path.join(save_dir, '[Figure 6] creativity and number of new edges'))
    plt.close()

def creativity_and_number_of_new_edges_update(df, save_dir=None):

    g = sns.JointGrid(data=df, x="centrality", y="new_edges", marginal_ticks=True)

    # Set a log scaling on the y axis
    g.ax_joint.set(xscale="log")
    #
    # g.plot_joint(sns.scatterplot, s=50, alpha=.5)
    # g.plot_marginals(sns.histplot, kde=True)

    g.plot_joint(sns.scatterplot, s=70, alpha=.6, color='#FC7F77', marker='p')
    # g.plot_marginals(sns.histplot, kde=True)
    sns.histplot(x=df['centrality'], fill=True, linewidth=1, ax=g.ax_marg_x, kde=True, color='#FC7F77')
    sns.histplot(y=df['new_edges'], fill=True, linewidth=0.5, ax=g.ax_marg_y, binwidth=2, discrete=True,
                 kde=True, line_kws={'color': 'k', 'linewidth': 1},
                 color='#FC7F77')
    g.ax_joint.set_xlabel('Creativity', fontsize=15)
    g.ax_joint.set_ylabel('Number of new edges', fontsize=15)
    plt.savefig(os.path.join(save_dir, '[Figure 6_update] creativity and number of new edges'))
    plt.close()


def degree_distribution(df, dataInst, save_dir=None):
    # degree distribution of melodic networks
    fig, ax = plt.subplots(figsize=(9,5.5))

    degree_mean_prob = {}
    for song in df['songID']:
        g = generate_network(dataInst, song, directed=False)
        degree = [d for (n,d) in g.degree()]
        degree_count = dict(collections.Counter(degree))
        degrees = list(degree_count.keys())
        degree_prob = [degree_count[d]/sum(degree_count.values()) for d in degrees]
        for (deg, prob) in zip(degrees, degree_prob):
            if deg not in degree_mean_prob:
                degree_mean_prob[deg] = []
            degree_mean_prob[deg].append(prob)

    X = list(degree_mean_prob.keys())
    Y = [np.mean(degree_mean_prob[x]) for x in X]
    Y_std = [np.std(degree_mean_prob[x]) for x in X]
    data_tuples = list(zip(X,Y,Y_std))
    data_tuples = sorted(data_tuples, key=lambda x: x[0])
    X = [d[0] for d in data_tuples]
    Y = np.array([d[1] for d in data_tuples])
    Y_std = np.array([d[2] for d in data_tuples])
    plt.plot(X, Y, color='darkorange', marker='o', linestyle='--', linewidth=0.5, markersize=3)
    plt.fill_between(X, Y - (Y_std/2), Y + (Y_std/2), color='moccasin', alpha=0.6)

    plt.xlabel('k', fontsize=21, fontstyle='italic')
    plt.ylabel('P(k)', fontsize=21, fontstyle='italic')
    plt.grid(linestyle='--')
    plt.savefig(os.path.join(save_dir, '[Figure 7] degree distribution'))
    plt.close()


def CC_by_song_length(less_creative, high_creative, X_bins, save_dir=None):
    fig, ax = plt.subplots(figsize=(9,5.5))
    bp1 = plt.boxplot(
        [less_creative[(less_creative['song_len']>=10**X_bins[i]) & (less_creative['song_len']<10**X_bins[i+1])]['cc'] for i in range(len(X_bins)-1)],
                positions=[1,4,7,10,13], widths=0.7, patch_artist=True, boxprops=dict(facecolor='purple', color='purple'), medianprops=dict(color='white'))
    bp2 = plt.boxplot(
        [high_creative[(high_creative['song_len']>=10**X_bins[i]) & (high_creative['song_len']<10**X_bins[i+1])]['cc'] for i in range(len(X_bins)-1)],
                positions=[2,5,8,11,14], widths=0.7, patch_artist=True, boxprops=dict(facecolor='red', color='red'), medianprops=dict(color='white'))
    plt.xlabel(r'$\log_{10} {song\ length}$', fontsize=15, fontstyle='italic')
    plt.ylabel('Clustering coefficient', fontsize=15, fontstyle='italic')
    plt.ylim(0,1) # cc range
    plt.grid(linestyle='--')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['less creative', 'highly creative'], loc='upper right', fontsize=13)
    plt.xticks([0, 3, 6, 9, 12, 15], np.round_(X_bins,2))
    plt.savefig(os.path.join(save_dir, '[Figure 8] CC by song length'))
    plt.close()


def ASPL_by_song_length(less_creative, high_creative, X_bins, save_dir=None):
    fig, ax = plt.subplots(figsize=(9,5.5))
    bp1 = plt.boxplot(
        [less_creative[(less_creative['song_len']>=10**X_bins[i]) & (less_creative['song_len']<10**X_bins[i+1])]['aspl'] for i in range(len(X_bins)-1)],
                positions=[1,4,7,10,13], widths=0.7, patch_artist=True, boxprops=dict(facecolor='purple', color='purple'), medianprops=dict(color='white'))
    bp2 = plt.boxplot(
        [high_creative[(high_creative['song_len']>=10**X_bins[i]) & (high_creative['song_len']<10**X_bins[i+1])]['aspl'] for i in range(len(X_bins)-1)],
                positions=[2,5,8,11,14], widths=0.7, patch_artist=True, boxprops=dict(facecolor='red', color='red'), medianprops=dict(color='white'))
    plt.xlabel(r'$\log_{10} {song\ length}$', fontsize=15, fontstyle='italic')
    plt.ylabel('Average shortest path length', fontsize=15, fontstyle='italic')
    plt.ylim(1,4) # aspl range
    plt.grid(linestyle='--')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['less creative', 'highly creative'], loc='upper right', fontsize=13)
    plt.xticks([0, 3, 6, 9, 12, 15], np.round_(X_bins,2))
    plt.savefig(os.path.join(save_dir, '[Figure 9] ASPL by song length'))
    plt.close()


def CC_by_era(less_creative, high_creative, eras, save_dir=None):
    fig, ax = plt.subplots(figsize=(9,5.5))
    bp1 = plt.boxplot(
        [less_creative[less_creative['era'].str.contains(era)]['cc'] for era in eras],
                positions=[1,4,7,10,13,16], widths=0.7, patch_artist=True, boxprops=dict(facecolor='purple', color='purple'), medianprops=dict(color='white'))
    bp2 = plt.boxplot(
        [high_creative[high_creative['era'].str.contains(era)]['cc'] for era in eras],
                positions=[2,5,8,11,14,17], widths=0.7, patch_artist=True, boxprops=dict(facecolor='red', color='red'), medianprops=dict(color='white'))
    plt.xlabel('Era', fontsize=15, fontstyle='italic')
    plt.ylabel('Clustering coefficient', fontsize=15, fontstyle='italic')
    plt.ylim(0,1) # cc range
    plt.grid(linestyle='--')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['less creative', 'highly creative'], loc='upper right', fontsize=13)
    plt.xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5], eras)
    plt.savefig(os.path.join(save_dir, '[Figure 10] era and cc_boxplot'))
    plt.close()


def ASPL_by_era(less_creative, high_creative, eras, save_dir=None):
    fig, ax = plt.subplots(figsize=(9,5.5))
    bp1 = plt.boxplot(
        [less_creative[less_creative['era'].str.contains(era)]['aspl'] for era in eras],
                positions=[1,4,7,10,13,16], widths=0.7, patch_artist=True, boxprops=dict(facecolor='purple', color='purple'), medianprops=dict(color='white'))
    bp2 = plt.boxplot(
        [high_creative[high_creative['era'].str.contains(era)]['aspl'] for era in eras],
                positions=[2,5,8,11,14,17], widths=0.7, patch_artist=True, boxprops=dict(facecolor='red', color='red'), medianprops=dict(color='white'))
    plt.xlabel('Era', fontsize=15, fontstyle='italic')
    plt.ylabel('Average shortest path length', fontsize=15, fontstyle='italic')
    plt.ylim(1,4) # aspl range
    plt.grid(linestyle='--')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['less creative', 'highly creative'], loc='upper right', fontsize=13)
    plt.xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5], eras)
    plt.savefig(os.path.join(save_dir, '[Figure 11] era and aspl_boxplot'))
    plt.close()


def MOD_by_song_length(less_creative, high_creative, X_bins, save_dir=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp1 = plt.boxplot(
        [less_creative[(less_creative['song_len'] >= 10 ** X_bins[i]) & (less_creative['song_len'] < 10 ** X_bins[i + 1])][
             'modularity'] for i in range(len(X_bins) - 1)],
        positions=[1, 4, 7, 10, 13], widths=0.7, patch_artist=True, boxprops=dict(facecolor='purple', color='purple'),
        medianprops=dict(color='white'))
    bp2 = plt.boxplot(
        [high_creative[
             (high_creative['song_len'] >= 10 ** X_bins[i]) & (high_creative['song_len'] < 10 ** X_bins[i + 1])][
             'modularity'] for i in range(len(X_bins) - 1)],
        positions=[2, 5, 8, 11, 14], widths=0.7, patch_artist=True, boxprops=dict(facecolor='red', color='red'),
        medianprops=dict(color='white'))
    plt.xlabel(r'$\log_{10} {song\ length}$', fontsize=15, fontstyle='italic')
    plt.ylabel('Modulraity', fontsize=15, fontstyle='italic')
    plt.ylim(0, 1)  # mod range
    plt.grid(linestyle='--')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['less creative', 'highly creative'], loc='upper right', fontsize=13)
    plt.xticks([0, 3, 6, 9, 12, 15], np.round_(X_bins, 2))
    plt.savefig(os.path.join(save_dir, '[Figure 12] modularity by song length'))
    plt.close()


def MOD_by_era(less_creative, high_creative, eras, save_dir=None):
    fig, ax = plt.subplots(figsize=(9,5.5))
    bp1 = plt.boxplot(
        [less_creative[less_creative['era'].str.contains(era)]['modularity'] for era in eras],
                positions=[1,4,7,10,13,16], widths=0.7, patch_artist=True, boxprops=dict(facecolor='purple', color='purple'), medianprops=dict(color='white'))
    bp2 = plt.boxplot(
        [high_creative[high_creative['era'].str.contains(era)]['modularity'] for era in eras],
                positions=[2,5,8,11,14,17], widths=0.7, patch_artist=True, boxprops=dict(facecolor='red', color='red'), medianprops=dict(color='white'))
    plt.xlabel('Era', fontsize=15, fontstyle='italic')
    plt.ylabel('Modularity', fontsize=15, fontstyle='italic')
    plt.ylim(0,1)  # mod range
    plt.grid(linestyle='--')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['less creative', 'highly creative'], loc='upper right', fontsize=13)
    plt.xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5], eras)
    plt.savefig(os.path.join(save_dir, '[Figure 13] modularity by era'))
    plt.close()


def main_result_figure(less_creative, high_creative, eras, save_dir=None):
    from scipy.stats import levene, ttest_ind, normaltest

    fig, axes = plt.subplots(1, 3, figsize=(6,5.5))

    # CC by Era
    less_creative_cc_by_era = [less_creative[less_creative['era'].str.contains(era)]['cc'] for era in eras]
    high_creative_cc_by_era = [high_creative[high_creative['era'].str.contains(era)]['cc'] for era in eras]
    print("-------------- CC t-test ---------------")
    for (lc_era, hc_era) in zip(less_creative_cc_by_era, high_creative_cc_by_era):
        _, norm_p = normaltest(lc_era)
        print(norm_p>0.1)
        _, norm_p = normaltest(hc_era)
        print(norm_p > 0.1)
        _, lresult = levene(lc_era, hc_era)
        if lresult < 0.05:
            t, p_value = ttest_ind(lc_era, hc_era, equal_var=False)
            print("levene: {:3f}, t statistic: {:.3f}, p-value: {:.3f}".format(lresult, t, p_value))
            print("num samples: low-{}, high-{}".format(len(lc_era),len(hc_era)))
        else:
            t, p_value = ttest_ind(lc_era, hc_era, equal_var=True)
            print("levene: {:3f}, t statistic: {:.3f}, p-value: {:.3f}".format(lresult, t, p_value))
            print("num samples: low-{}, high-{}".format(len(lc_era), len(hc_era)))

    axes[0].boxplot(
        less_creative_cc_by_era,
                positions=[1,4,7,10,13,16], widths=0.7, patch_artist=True, boxprops=dict(facecolor='purple', color='purple'), medianprops=dict(color='white'))
    axes[0].boxplot(
        high_creative_cc_by_era,
                positions=[2,5,8,11,14,17], widths=0.7, patch_artist=True, boxprops=dict(facecolor='red', color='red'), medianprops=dict(color='white'))
    plt.ylim(0, 1)  # cc range
    axes[0].grid(linestyle='--')
    axes[0].set_xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5])
    axes[0].set_xticklabels(['B','C','R','P','M','T'], fontsize=20, fontfamily='serif')
    axes[0].set_ylabel('CC', fontsize=28, fontfamily='serif')
    axes[0].yaxis.set_tick_params(labelsize=18)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes[0].spines[axis].set_linewidth(2.0)

    # ASPL by Era
    less_creative_aspl_by_era = [less_creative[less_creative['era'].str.contains(era)]['aspl'] for era in eras]
    high_creative_aspl_by_era = [high_creative[high_creative['era'].str.contains(era)]['aspl'] for era in eras]
    print("-------------- ASPL t-test ---------------")
    for (lc_era, hc_era) in zip(less_creative_aspl_by_era, high_creative_aspl_by_era):
        _, norm_p = normaltest(lc_era)
        print(norm_p > 0.1)
        _, norm_p = normaltest(hc_era)
        print(norm_p > 0.1)
        _, lresult = levene(lc_era, hc_era)
        if lresult < 0.05:
            t, p_value = ttest_ind(lc_era, hc_era, equal_var=False)
            print("levene: {:3f}, t statistic: {:.3f}, p-value: {:.3f}".format(lresult, t, p_value))
            print("num samples: low-{}, high-{}".format(len(lc_era), len(hc_era)))
        else:
            t, p_value = ttest_ind(lc_era, hc_era, equal_var=True)
            print("levene: {:3f}, t statistic: {:.3f}, p-value: {:.3f}".format(lresult, t, p_value))
            print("num samples: low-{}, high-{}".format(len(lc_era), len(hc_era)))

    axes[1].boxplot(
        less_creative_aspl_by_era,
        positions=[1, 4, 7, 10, 13, 16], widths=0.7, patch_artist=True,
        boxprops=dict(facecolor='purple', color='purple'), medianprops=dict(color='white'))
    axes[1].boxplot(
        high_creative_aspl_by_era,
        positions=[2, 5, 8, 11, 14, 17], widths=0.7, patch_artist=True, boxprops=dict(facecolor='red', color='red'),
        medianprops=dict(color='white'))
    #plt.ylim(1, 4)  # aspl range
    axes[1].grid(linestyle='--')
    axes[1].set_xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5])
    axes[1].set_xticklabels(['B', 'C', 'R', 'P', 'M', 'T'], fontsize=20, fontfamily='serif')
    axes[1].set_ylabel('ASPL', fontsize=28, fontfamily='serif')
    axes[1].yaxis.set_tick_params(labelsize=18)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes[1].spines[axis].set_linewidth(2.0)

    # Mod by Era
    less_creative_mod_by_era = [less_creative[less_creative['era'].str.contains(era)]['modularity'] for era in eras]
    high_creative_mod_by_era = [high_creative[high_creative['era'].str.contains(era)]['modularity'] for era in eras]
    print("-------------- MOD t-test ---------------")
    for (lc_era, hc_era) in zip(less_creative_mod_by_era, high_creative_mod_by_era):
        _, norm_p = normaltest(lc_era)
        print(norm_p > 0.1)
        _, norm_p = normaltest(hc_era)
        print(norm_p > 0.1)
        _, lresult = levene(lc_era, hc_era)
        if lresult < 0.05:
            t, p_value = ttest_ind(lc_era, hc_era, equal_var=False)
            print("levene: {:3f}, t statistic: {:.3f}, p-value: {:.3f}".format(lresult, t, p_value))
            print("num samples: low-{}, high-{}".format(len(lc_era), len(hc_era)))
        else:
            t, p_value = ttest_ind(lc_era, hc_era, equal_var=True)
            print("levene: {:3f}, t statistic: {:.3f}, p-value: {:.3f}".format(lresult, t, p_value))
            print("num samples: low-{}, high-{}".format(len(lc_era), len(hc_era)))

    axes[2].boxplot(
        less_creative_mod_by_era,
                positions=[1,4,7,10,13,16], widths=0.7, patch_artist=True, boxprops=dict(facecolor='purple', color='purple'), medianprops=dict(color='white'))
    axes[2].boxplot(
        high_creative_mod_by_era,
                positions=[2,5,8,11,14,17], widths=0.7, patch_artist=True, boxprops=dict(facecolor='red', color='red'), medianprops=dict(color='white'))
    plt.ylim(0, 1)  # mod range
    axes[2].grid(linestyle='--')
    axes[2].set_xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5])
    axes[2].set_xticklabels(['B', 'C', 'R', 'P', 'M', 'T'], fontsize=20, fontfamily='serif')
    axes[2].set_ylabel('MOD', fontsize=28, fontfamily='serif')
    axes[2].yaxis.set_tick_params(labelsize=18)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes[2].spines[axis].set_linewidth(2.0)

    #fig.text(0.5, 0.001, 'Musical era', ha='center', fontsize=20, fontfamily='serif')
    plt.subplots_adjust(wspace=0.6)
    plt.tight_layout()

    # CC by song length
    # axes[1,0].boxplot(
    #     [less_creative[(less_creative['song_len']>=X_bins_log[i]) & (less_creative['song_len']<X_bins_log[i+1])]['cc'] for i in range(len(X_bins_log)-1)],
    #             positions=[26,233,2468], widths=[15,140,1400], patch_artist=True, boxprops=dict(facecolor='purple', color='purple'), medianprops=dict(color='white'))
    # axes[1,0].boxplot(
    #     [high_creative[(high_creative['song_len']>=X_bins_log[i]) & (high_creative['song_len']<X_bins_log[i+1])]['cc'] for i in range(len(X_bins_log)-1)],
    #             positions=[72,542,5217], widths=[30,300,2500], patch_artist=True, boxprops=dict(facecolor='red', color='red'), medianprops=dict(color='white'))
    # plt.ylim(0, 1)  # cc range
    # plt.grid(linestyle='--')
    # axes[1,0].set_xlim([9, 10001])
    # axes[1,0].set_xscale("log")
    # x_major = matplotlib.ticker.LogLocator(base=10.0, numticks=5)
    # axes[1,0].xaxis.set_major_locator(x_major)
    # x_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    # axes[1,0].xaxis.set_minor_locator(x_minor)
    # axes[1,0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     axes[1,0].spines[axis].set_linewidth(1.3)
    # axes[1,0].set_ylabel('Song Length', fontsize=20, fontfamily='serif')
    # axes[1,0].xaxis.set_tick_params(labelsize=16)
    # #axes[1,0].set_xticks([0, 3, 6, 9, 12, 15])
    # #axes[1,0].set_xticklabels(np.round_(X_bins, 2))
    #
    # # ASPL by song length
    # axes[1,1].boxplot(
    #     [less_creative[
    #          (less_creative['song_len'] >= X_bins_log[i]) & (less_creative['song_len'] < X_bins_log[i+1])]['aspl']
    #      for i in range(len(X_bins_log)-1)],
    #     positions=[26,233,2468], widths=[15,140,1400], patch_artist=True, boxprops=dict(facecolor='purple', color='purple'),
    #     medianprops=dict(color='white'))
    # axes[1,1].boxplot(
    #     [high_creative[
    #          (high_creative['song_len'] >= X_bins_log[i]) & (high_creative['song_len'] < X_bins_log[i+1])]['aspl']
    #      for i in range(len(X_bins_log)-1)],
    #     positions=[72,542,5217], widths=[30,300,2500], patch_artist=True, boxprops=dict(facecolor='red', color='red'),
    #     medianprops=dict(color='white'))
    # plt.ylim(1, 4)  # aspl range
    # plt.grid(linestyle='--')
    # axes[1,1].set_xlim([9, 10001])
    # axes[1,1].set_xscale("log")
    # x_major = matplotlib.ticker.LogLocator(base=10.0, numticks=5)
    # axes[1,1].xaxis.set_major_locator(x_major)
    # x_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    # axes[1,1].xaxis.set_minor_locator(x_minor)
    # axes[1,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     axes[1,1].spines[axis].set_linewidth(1.3)
    # axes[1,1].xaxis.set_tick_params(labelsize=16)
    # #axes[1,1].set_xticks([0, 3, 6, 9, 12, 15])
    # #axes[1,1].set_xticklabels(np.round_(X_bins, 2))
    #
    # # Mod by song length
    # axes[1,2].boxplot(
    #     [less_creative[
    #          (less_creative['song_len'] >= X_bins_log[i]) & (less_creative['song_len'] < X_bins_log[i+1])][
    #          'modularity'] for i in range(len(X_bins_log)-1)],
    #     positions=[26,233,2468], widths=[15,140,1400], patch_artist=True, boxprops=dict(facecolor='purple', color='purple'),
    #     medianprops=dict(color='white'))
    # axes[1,2].boxplot(
    #     [high_creative[
    #          (high_creative['song_len'] >= X_bins_log[i]) & (high_creative['song_len'] < X_bins_log[i+1])][
    #          'modularity'] for i in range(len(X_bins_log)-1)],
    #     positions=[72,542,5217], widths=[30,300,2500], patch_artist=True, boxprops=dict(facecolor='red', color='red'),
    #     medianprops=dict(color='white'))
    # plt.ylim(0, 1)  # mod range
    # plt.grid(linestyle='--')
    # axes[1,2].set_xlim([9, 10001])
    # axes[1,2].set_xscale("log")
    # x_major = matplotlib.ticker.LogLocator(base=10.0, numticks=5)
    # axes[1,2].xaxis.set_major_locator(x_major)
    # x_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    # axes[1,2].xaxis.set_minor_locator(x_minor)
    # axes[1,2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     axes[1,2].spines[axis].set_linewidth(1.3)
    # axes[1,2].xaxis.set_tick_params(labelsize=16)
    # #axes[1,2].set_xticks([0, 3, 6, 9, 12, 15])
    # #axes[1,2].set_xticklabels(np.round_(X_bins, 2))

    plt.savefig(os.path.join(save_dir, '[Figure 14] Main Result_2'))
    plt.close()



def power_law_fitting_with_jacknife_error(xdata, ydata):
    def power_law(x,a,b):
        return a * np.power(x,b)
    leave_one_out_param = []
    for i in range(len(xdata)):
        x = copy.deepcopy(xdata)
        y = copy.deepcopy(ydata)
        x.remove(xdata[i])
        y.remove(ydata[i])
        popt, pcov = curve_fit(power_law, x, y)
        leave_one_out_param.append(popt[1])
    leave_one_out_param = np.array(leave_one_out_param)
    mean_leave_one_out_param = np.mean(leave_one_out_param)
    jacknife_variance = ((len(xdata)-1)/len(xdata)) * sum(np.power((leave_one_out_param - mean_leave_one_out_param), 2))
    jacknife_error = math.sqrt(jacknife_variance)
    return mean_leave_one_out_param, jacknife_error


def pearson_jacknife(xdata, ydata):
    leave_one_out_param = []
    for i in range(len(xdata)):
        x = copy.deepcopy(xdata)
        y = copy.deepcopy(ydata)
        x.remove(xdata[i])
        y.remove(ydata[i])
        r, _ = pearsonr(x,y)
        leave_one_out_param.append(r)
    leave_one_out_param = np.array(leave_one_out_param)
    mean_leave_one_out_param = np.mean(leave_one_out_param)
    jacknife_variance = ((len(xdata)-1)/len(xdata)) * sum(np.power((leave_one_out_param - mean_leave_one_out_param), 2))
    jacknife_error = math.sqrt(jacknife_variance)
    return mean_leave_one_out_param, jacknife_error



def creativity_plot2(df, save_dir=None):
    sns.set_theme(style="white")

    # Plot miles per gallon against horsepower with other semantics
    sns.relplot(x="year", y="centrality", hue="era",
                sizes=(40, 400), alpha=.5, palette="muted",
                height=6, data=df)
    plt.savefig(os.path.join(save_dir, 'Creativity_plot2.png'))


def creativity_and_entropy(df, save_dir=None):
    fig, ax = plt.subplots()
    plt.scatter(df['centrality'], df['melodic_entropy'], marker='o', facecolors='hotpink', edgecolors='black', s=9.)
    #plt.xscale('log')
    #plt.xlabel(r'$log_{10}{creativity}$', fontsize=15)
    plt.xlabel('Creativity', fontsize=15)
    plt.ylabel('Melodic Entropy', fontsize=15)
    plt.savefig(os.path.join(save_dir, 'Creativity and Melodic Entropy'))
    plt.close()


def creativity_and_diversity(df, save_dir=None):
    fig, ax = plt.subplots()
    plt.scatter(df['centrality'], df['melodic_diversity'], marker='o', facecolors='hotpink', edgecolors='black', s=9.)
    #plt.xscale('log')
    #plt.xlabel(r'$log_{10}{creativity}$', fontsize=15)
    plt.xlabel('Creativity', fontsize=15)
    plt.ylabel('Melodic Diversity', fontsize=15)
    plt.savefig(os.path.join(save_dir, 'Creativity and Melodic Diversity'))
    plt.close()


def creativity_and_consonance(df, save_dir=None):
    fig, ax = plt.subplots()
    plt.scatter(df['centrality'], df['melodic_consonance'], marker='o', facecolors='hotpink', edgecolors='black', s=9.)
    #plt.xscale('log')
    #plt.xlabel(r'$log_{10}{creativity}$', fontsize=15)
    plt.xlabel('Creativity', fontsize=15)
    plt.ylabel('Melodic Consonance', fontsize=15)
    plt.savefig(os.path.join(save_dir, 'Creativity and Melodic Consonance'))
    plt.close()

def creativity_and_average_pitch_interval(df, save_dir=None):
    fig, ax = plt.subplots()
    plt.scatter(df['centrality'], df['average_pitch_interval'], marker='o', facecolors='hotpink', edgecolors='black', s=9.)
    #plt.xscale('log')
    #plt.xlabel(r'$log_{10}{creativity}$', fontsize=15)
    plt.xlabel('Creativity', fontsize=15)
    plt.ylabel('Average Pitch Interval', fontsize=15)
    plt.savefig(os.path.join(save_dir, 'Creativity and Average Pitch Interval'))
    plt.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Drawing figures')
    parser.add_argument('--element', type=str, required=True, help='which element to encode data, rhythm or melody')
    parser.add_argument('--songID', type=str, default='Bach103', help='which song to draw Markov matrix of melodic bigrams')
    parser.add_argument('--enlarge', type=str, default='False',
                        help='whether to enlarge the figure of Markov matrix of melodic bigrams')
    parser.add_argument('--low', type=int, default=conf.LOW,
                        help='number of less creative songs for comparison')
    parser.add_argument('--high', type=int, default=conf.HIGH,
                        help='number of highly creative songs for comparison')
    args = parser.parse_args()

    figure_dir = None
    analysis_filename = None
    weighted_edge_filename = None
    if args.element == 'melody':
        figure_dir = conf.figure_melody_dir
        analysis_filename = conf.analysis_filename
        weighted_edge_filename = conf.similarity_filename
    elif args.element == 'rhythm':
        figure_dir = conf.figure_rhythm_dir
        analysis_filename = conf.analysis_filename_rhythm
        weighted_edge_filename = conf.similarity_rhythm_filename

    dataInst = load_data(element=args.element)

    df = pd.read_excel(os.path.join(conf.result_dir, analysis_filename), index_col=0)
    less_creative = df.sort_values(by=['centrality'], axis=0).iloc[:600, :]
    high_creative = df.sort_values(by=['centrality'], axis=0).iloc[-600:, :]

    #X_bins = np.linspace(math.log10(min(df['song_len'])), math.log10(max(df['song_len'])), conf.song_length_bin)
    X_bins_log = [10,100,1000,10000]

    with open(os.path.join(conf.result_dir, weighted_edge_filename), "rb") as f:
       weighted_edge = dill.load(f)

    year_sorted_songs = list(df.sort_values(by=['year'], ascending=True)['songID'])

    #Figure 1(A)
    #pdf_of_bigrams(df, dataInst, element=args.element, save_dir=figure_dir)

    #Figure 1(B)
    #histogram_of_edge_weight(weighted_edge, save_dir=figure_dir)
    #edge_weight_heatmap(df, year_sorted_songs, weighted_edge, save_dir=figure_dir)

    #Figure 1(C)
    #song_representation(args.songID, save_dir=figure_dir)
    #song_representation_enlarge(args.songID, save_dir=figure_dir)

    #Figure 3
    #creativity_plot(df, save_dir=figure_dir)
    #creativity_plot2(df, save_dir=figure_dir)
    #creative_network(dataInst, songID=args.songID, save_dir=figure_dir)

    #Figure 4(A)
    #creativity_and_number_of_new_nodes(df, save_dir=figure_dir)
    #creativity_and_number_of_new_nodes_update(df, save_dir=figure_dir)

    #Figure 4(B)
    #creativity_and_number_of_new_edges(df, save_dir=figure_dir)
    #creativity_and_number_of_new_edges_update(df, save_dir=figure_dir)

    #Figure 5
    #degree_distribution(df, dataInst, save_dir=figure_dir)

    #Figure 6(A)
    #CC_by_song_length(less_creative, high_creative, X_bins)

    #Figure 6(B)
    #ASPL_by_song_length(less_creative, high_creative, X_bins)

    #Figure 7(A)
    #CC_by_era(less_creative, high_creative, eras, save_dir=figure_dir)

    #Figure 7(B)
    #ASPL_by_era(less_creative, high_creative, eras, save_dir=figure_dir)

    #Figure 8(A)
    #MOD_by_song_length(less_creative, high_creative, X_bins)

    #Figure 8(B)
    #MOD_by_era(less_creative, high_creative, eras, save_dir=figure_dir)

    # Main result figure
    main_result_figure(less_creative, high_creative, eras=conf.eras, save_dir=figure_dir)


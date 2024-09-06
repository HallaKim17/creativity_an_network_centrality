from read_data import ReadDataInfo
import pandas as pd
from network_utils import *
from helpers import *
import numpy as np
import argparse
import collections
import math
import pretty_midi
import re


def make_dataframe(songs, dataInst, dataInfo, creativity, element):
    df = pd.DataFrame(
        columns=['songID', 'composer', 'year', 'era', 'centrality', 'cc', 'aspl', 'rand_cc', 'rand_aspl',
                 'song_len', 'density', 'modularity', 'new_nodes', 'new_edges'])
    df['songID'] = songs
    composer = []
    year = []
    era = []
    crea = []
    mod = []
    for i,song in enumerate(songs):
        print(i)
        composer.append(dataInfo[dataInfo['songID'] == song]['composer'].values[0])
        year.append(int(dataInfo[dataInfo['songID'] == song]['year']))
        era.append(dataInfo[dataInfo['songID'] == song]['era'].values[0])
        crea.append(creativity[song])
        mod.append(get_modularity(dataInst, song, element=element)) # 리듬의 경우, 곡 전체가 한 가지 비트로 이루어져 있으면(노드가 하나) modularity 0으로 둠.
    df['composer'] = composer
    df['year'] = year
    df['era'] = era
    df['centrality'] = crea
    df['modularity'] = mod
    # df['composer'] = [dataInfo[dataInfo['songID'] == song]['composer'].values[0] for song in songs]
    # df['year'] = [int(dataInfo[dataInfo['songID'] == song]['year']) for song in songs]
    # df['era'] = [dataInfo[dataInfo['songID'] == song]['era'].values[0] for song in songs]
    # df['centrality'] = [creativity[song] for song in songs]
    # df['modularity'] = [get_modularity(dataInst, song, element=element) for song in songs] # 리듬의 경우, 곡 전체가 한 가지 비트로 이루어져 있으면(노드가 하나) modularity 0으로 둠.
    if element=='melody':
        df['song_len'] = [len(dataInst[song].melody()) for song in songs]
        df['density'] = [len(set(dataInst[song].bigram())) / (
                (len(set(dataInst[song].melody())) * (len(set(dataInst[song].melody())) - 1)) / 2) for song in songs]
        df['aspl'] = [get_aspl(generate_network(dataInst=dataInst, songID=song, directed=False, element=element)) for
                      song in songs]
    elif element=='rhythm':
        df['song_len'] = [len(dataInst[song].rhythm()) for song in songs]
        dens = []
        aspl = []
        for i,song in enumerate(songs):
            print('density and aspl ', i)
            if len(set(dataInst[song].rhythm())) == 1:
                dens.append(1) # 리듬의 경우, 곡 전체가 한 가지 비트로 이루어져 있으면(노드가 하나) density 1으로 둠.
                aspl.append(1) # 리듬의 경우, 곡 전체가 한 가지 비트로 이루어져 있으면(노드가 하나) aspl 1으로 둠.
            else:
                dens.append(len(set(dataInst[song].rhythm_bigram())) / ((len(set(dataInst[song].rhythm())) * (len(set(dataInst[song].rhythm())) - 1)) / 2))
                aspl.append(get_aspl(generate_network(dataInst=dataInst, songID=song, directed=False, element=element)))
        df['density'] = dens
        df['aspl'] = aspl
    df['cc'] = [get_cc(generate_network(dataInst=dataInst, songID=song, directed=False, element=element)) for song in songs]

    return df


# Melody nodes
def count_new_nodes(songID, df, dataInst):
    this_song_year = df[df['songID'] == songID]['year'].values[0]
    previous_songs = df[df['year'] < this_song_year]['songID']
    already_used_notes = []
    for song in previous_songs:
        notes = list(set(dataInst[song].melody()))
        already_used_notes.append(notes)
    already_used_notes = set(sum(already_used_notes, []))
    this_song_notes = set(dataInst[songID].melody())
    new_notes_in_this_song = this_song_notes - already_used_notes
    return len(new_notes_in_this_song)

# Melody edges
def count_new_edges(songID, df, dataInst):
    this_song_year = df[df['songID'] == songID]['year'].values[0]
    previous_songs = df[df['year'] < this_song_year]['songID']

    already_used_notes = []
    for song in previous_songs:
        notes = list(set(dataInst[song].melody()))
        already_used_notes.append(notes)
    already_used_notes = set(sum(already_used_notes, []))

    already_used_bigrams = []
    for song in previous_songs:
        bigrams = list(set(dataInst[song].bigram()))
        already_used_bigrams.append(bigrams)
    already_used_bigrams = set(sum(already_used_bigrams, []))
    this_song_bigrams = set(dataInst[songID].bigram())
    new_bigrams_in_this_song = this_song_bigrams - already_used_bigrams
    new_bigrams_with_previous_notes_in_this_song = []
    for (note1, note2) in new_bigrams_in_this_song:
        if (note1 in already_used_notes) & (note2 in already_used_notes):
            new_bigrams_with_previous_notes_in_this_song.append((note1,note2))
    return len(new_bigrams_with_previous_notes_in_this_song)

# Rhythm nodes
def count_new_nodes_rhythm(songID, df, dataInst):
    this_song_year = df[df['songID'] == songID]['year'].values[0]
    previous_songs = df[df['year'] < this_song_year]['songID']
    already_used_beats = []
    for song in previous_songs:
        beats = list(set(dataInst[song].rhythm()))
        already_used_beats.append(beats)
    already_used_beats = set(sum(already_used_beats, []))
    this_song_beats = set(dataInst[songID].rhythm())
    new_beats_in_this_song = this_song_beats - already_used_beats
    return len(new_beats_in_this_song)

# Rhythm edges
def count_new_edges_rhythm(songID, df, dataInst):
    this_song_year = df[df['songID'] == songID]['year'].values[0]
    previous_songs = df[df['year'] < this_song_year]['songID']

    already_used_rhythms = []
    for song in previous_songs:
        notes = list(set(dataInst[song].melody()))
        already_used_rhythms.append(notes)
    already_used_notes = set(sum(already_used_rhythms, []))

    already_used_bigrams = []
    for song in previous_songs:
        bigrams = list(set(dataInst[song].rhythm_bigram()))
        already_used_bigrams.append(bigrams)
    already_used_bigrams = set(sum(already_used_bigrams, []))
    this_song_bigrams = set(dataInst[songID].rhythm_bigram())
    new_bigrams_in_this_song = this_song_bigrams - already_used_bigrams
    new_bigrams_with_previous_beats_in_this_song = []
    for (beat1, beat2) in new_bigrams_in_this_song:
        if (beat1 in already_used_notes) & (beat2 in already_used_notes):
            new_bigrams_with_previous_beats_in_this_song.append((beat1,beat2))
    return len(new_bigrams_with_previous_beats_in_this_song)


# Pitch Count (PC), Pitch Count/Bar (PC/bar), Pitch Class Histogram (PCH), Pitch Class Transition Matrix (PCTM), Pitch Range (PR), average Pitch Interval (PI)
def melodic_entropy(songID, dataInst):
    melody = dataInst[songID].melody()
    count = collections.Counter(melody)
    prob = {k:v/sum(count.values()) for k,v in count.items()}
    entropy = sum([p * (1/math.log2(p)) for p in prob])
    return entropy

def melodic_diversity(songID, dataInst):
    return len(set(dataInst[songID].melody()))/len(dataInst[songID].melody())

def melodic_consonance(songID, dataInst):
    # 인접한 두 음(melodic bigram)의 consonance 값의 합을 melodic bigram의 총 개수로 나눈 것.
    pitch_class = {'C':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4, 'F':5, 'F#':6, 'Gb':6, 'G':7, 'G#':8, 'Ab':8, 'A':9, 'A#':10, 'Bb':10, 'B':11}
    consonance = []
    for (note1, note2) in dataInst[songID].bigram():
        pc1, pc2 = pretty_midi.note_number_to_name(note1), pretty_midi.note_number_to_name(note2)
        if '-' in pc1:
            pc1 = pc1.replace("-","")
        if '-' in pc2:
            pc2 = pc2.replace("-","")
        pc1 = re.sub(r'[0-9]+', '', pc1)
        pc2 = re.sub(r'[0-9]+', '', pc2)
        pc1, pc2 = pitch_class[pc1], pitch_class[pc2]
        interval = abs(pc1-pc2)
        if interval in [0,3,4,7,8,9]:
            consonance.append(1)
        elif interval == 5:
            consonance.append(0)
        else:
            consonance.append(-1)
    return sum(consonance) / len(consonance)

def average_pitch_interval(songID, dataInst):
    pitch_class = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
                   'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    intervals = []
    for (note1, note2) in dataInst[songID].bigram():
        pc1, pc2 = pretty_midi.note_number_to_name(note1), pretty_midi.note_number_to_name(note2)
        if '-' in pc1:
            pc1 = pc1.replace("-", "")
        if '-' in pc2:
            pc2 = pc2.replace("-", "")
        pc1 = re.sub(r'[0-9]+', '', pc1)
        pc2 = re.sub(r'[0-9]+', '', pc2)
        pc1, pc2 = pitch_class[pc1], pitch_class[pc2]
        interval = abs(pc1 - pc2)
        intervals.append(interval)
    return sum(intervals) / len(intervals)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='data analysis')
    parser.add_argument('--element', type=str, required=True, help='which element to encode data, rhythm or melody')
    args = parser.parse_args()

    dataInst = load_data(args.element)
    dataInfo = ReadDataInfo(dataInfo_dir=conf.dataInfo_dir)
    songs = list(dataInst.keys())

    creativity_score_filename = None
    analysis_filename = None
    if args.element == 'melody':
        creativity_score_filename = conf.creativity_score_filename
        analysis_filename = conf.analysis_filename
    elif args.element == 'rhythm':
        creativity_score_filename = conf.creativity_score_rhythm_filename
        analysis_filename = conf.analysis_filename_rhythm

    with open(os.path.join(conf.result_dir, creativity_score_filename), "rb") as f:
        creativity = dill.load(f)

    # make dataframe having features of melodic networks of all songs
    print("================ Making data for analysis ================")
    df = make_dataframe(songs, dataInst, dataInfo, creativity, element=args.element)

    # Add columns of number of new nodes(notes) and edges(bigrams) of each song
    new_nodes = []
    for i, song in enumerate(df['songID']):
        print('count new nodes ', i)
        if args.element=='melody':
            new_nodes.append(count_new_nodes(song, df, dataInst))
        elif args.element=='rhythm':
            new_nodes.append(count_new_nodes_rhythm(song, df, dataInst))
    df['new_nodes'] = new_nodes

    new_edges = []
    for i, song in enumerate(df['songID']):
        print('count new edges ', i)
        if args.element=='melody':
            new_edges.append(count_new_edges(song, df, dataInst))
        elif args.element=='rhythm':
            new_edges.append(count_new_edges_rhythm(song, df, dataInst))
    df['new_edges'] = new_edges

    with pd.ExcelWriter(os.path.join(conf.result_dir, analysis_filename)) as writer:
        df.to_excel(writer)

    df = pd.read_excel(os.path.join(conf.result_dir, analysis_filename), index_col=0)

    # Add columns of random clustering coefficient and average shortest path length of random networks by switching algorithm.
    print("================ Generating random networks ================")
    num_networks = 10  # conf.number_of_random_networks
    rand_aspl = []
    rand_cc = []
    rand_mod = []
    for i in range(df.shape[0]):
        print(i, '/', df.shape[0])
        g = generate_network(dataInst, df['songID'][i], directed=True, element=args.element)
        g_ = generate_network(dataInst, df['songID'][i], directed=False, element=args.element)
        raspls = []  # random average shortest path lengths
        rccs = []  # random clustering coefficients
        rms = []  # random modularities
        for i in range(num_networks):
            r = generate_directed_random_networks_by_switching_algorithm(g)
            #r_aspl = nx.average_shortest_path_length(r)
            aspl = None
            if len(r.edges) == 1:
                r_aspl = 1
            else:
                r_aspl = get_aspl(r)
            raspls.append(r_aspl)
            r = generate_undirected_random_networks_by_switching_algorithm(g_)
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

    with pd.ExcelWriter(os.path.join(conf.result_dir, analysis_filename)) as writer:
        df.to_excel(writer)
    print("================ dataframe file saved ================")

    df = pd.read_excel(os.path.join(conf.result_dir, conf.analysis_filename), index_col=0)
    entropy = []
    diversity = []
    consonance = []
    pitch_interval = []
    for i in range(df.shape[0]):
        entropy.append(melodic_entropy(df['songID'][i], dataInst))
        diversity.append(melodic_diversity(df['songID'][i], dataInst))
        consonance.append(melodic_consonance(df['songID'][i], dataInst))
        pitch_interval.append(average_pitch_interval(df['songID'][i], dataInst))
        print(i)
    df['melodic_entropy'] = entropy
    df['melodic_diversity'] = diversity
    df['melodic_consonance'] = consonance
    df['average_pitch_interval'] = pitch_interval

    with pd.ExcelWriter(os.path.join(conf.result_dir, "data_analysis_1000_random_networks_supple_metrics.xlsx")) as writer:
        df.to_excel(writer)

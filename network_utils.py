import networkx as nx
import community
from networkx.algorithms.community import modularity
import random
import itertools


def generate_network(dataInst, songID, directed=False, element=None):
    if not directed:
        g = nx.Graph()
        if element=='melody':
            g.add_edges_from(list(set(dataInst[songID].bigram())))
        elif element=='rhythm':
            g.add_edges_from(list(set(dataInst[songID].rhythm_bigram())))
        return g
    else:
        g = nx.DiGraph()
        if element=='melody':
            g.add_edges_from(list(set(dataInst[songID].bigram())))
        elif element=='rhythm':
            g.add_edges_from(list(set(dataInst[songID].rhythm_bigram())))
        return g

def get_cc(g):
    # g should be undirected graph
    cc = nx.transitivity(g)  # fraction of triangles in the whole network (easy to interpret)
    return cc

def get_aspl(g, directed=True):
    path_len_reciprocal = 0  # 1/d_(i,j)

    if directed:
        for n1 in g.nodes:
            for n2 in g.nodes:
                if n1 != n2:
                    try:
                        path_len_reciprocal += (1 / nx.shortest_path_length(g, source=n1, target=n2))
                    except Exception as ex:
                        path_len_reciprocal += 0  # if no path exists, zero (reciprocal of infinity) is added.

        aspl = (path_len_reciprocal / (len(g.nodes) * (len(g.nodes) - 1))) ** (-1)
    else:
        for (n1,n2) in list(itertools.combinations(g.nodes,2)):
            try:
                path_len_reciprocal += (1 / nx.shortest_path_length(g, source=n1, target=n2))
            except Exception as ex:
                path_len_reciprocal += 0  # if no path exists, zero (reciprocal of infinity) is added.

        aspl = (path_len_reciprocal / ((len(g.nodes) * (len(g.nodes) - 1))/2)) ** (-1)

    return aspl


def get_modularity(dataInst, songID, element=None):
    g = generate_network(dataInst, songID, directed=False, element=element)
    partition = community.best_partition(g)
    community_nodes = []
    for com in sorted(list(set(partition.values()))):
        list_nodes = [node for node in partition.keys()
                      if partition[node] == com]
        community_nodes.append(list_nodes)
    return modularity(g,community_nodes)


def get_random_graph_modularity(g):
    partition = community.best_partition(g)
    community_nodes = []
    for com in sorted(list(set(partition.values()))):
        list_nodes = [node for node in partition.keys()
                      if partition[node] == com]
        community_nodes.append(list_nodes)
    return modularity(g,community_nodes)


def edge_table_for_undirected_graph(g):
    first_node_list = []
    second_node_list = []
    adj_list = {}
    for edge in map(str.split, nx.generate_adjlist(g)):
        this_node = int(edge[0])
        adj_list[this_node] = []
        other_nodes = list(map(int, edge[1:]))
        for other_node in other_nodes:
            adj_list[this_node].append(other_node)
            first_node_list.append(this_node)
            second_node_list.append(other_node)
    return first_node_list, second_node_list, adj_list


def edge_table_for_directed_graph(g):
    source_node_list = []
    target_node_list = []
    adj_list = {}
    for edge in map(str.split, nx.generate_adjlist(g)):
        this_node = int(edge[0])
        adj_list[this_node] = []
        other_nodes = list(map(int, edge[1:]))
        for other_node in other_nodes:
            adj_list[this_node].append(other_node)
            source_node_list.append(this_node)
            target_node_list.append(other_node)
    return source_node_list, target_node_list, adj_list


def undirected_random_net_by_switching_algorithm(g):
    first_node_list, second_node_list, adj_list = edge_table_for_undirected_graph(g)
    Q = 10
    E = len(first_node_list)

    for k in range(Q * E):
        i,j = random.sample(range(len(first_node_list)), 2)
        a = first_node_list[i]
        b = second_node_list[i]
        c = first_node_list[j]
        d = second_node_list[j]
        if len({a,b,c,d}) == 4:
            if (d not in adj_list[a]) & (a not in adj_list[d]):
                if (b not in adj_list[c]) & (c not in adj_list[b]):
                    second_node_list[i] = d
                    second_node_list[j] = b
                    adj_list[a].remove(b)
                    adj_list[a].append(d)
                    adj_list[c].remove(d)
                    adj_list[c].append(b)
                    #print((a,b),(c,d),'==>',(a,d),(c,b))
    return first_node_list, second_node_list, adj_list


def directed_random_net_by_switching_algorithm(g):
    source_node_list, target_node_list, adj_list = edge_table_for_directed_graph(g)
    Q = 10
    E = len(source_node_list)

    for k in range(Q * E):
        i,j = random.sample(range(len(source_node_list)), 2)
        a = source_node_list[i]
        b = target_node_list[i]
        c = source_node_list[j]
        d = target_node_list[j]
        if len({a,b,c,d}) == 4:
            if (d not in adj_list[a]) & (b not in adj_list[c]):
                target_node_list[i] = d
                target_node_list[j] = b
                adj_list[a].remove(b)
                adj_list[c].remove(d)
                adj_list[a].append(d)
                adj_list[c].append(b)
                #print((a,b),(c,d),'==>',(a,d),(c,b))
    return source_node_list, target_node_list, adj_list


def two_node_list_to_undirected_graph(small_node_list, big_node_list):
    edges = list(zip(small_node_list, big_node_list))
    g = nx.Graph()
    g.add_edges_from(edges)
    return g

def two_node_list_to_directed_graph(source_node_list, target_node_list):
    edges = list(zip(source_node_list, target_node_list))
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def generate_undirected_random_networks_by_switching_algorithm(g):
    """
    parameter g: melodic network of a song
    """
    # generate a single random network by with a given number of nodes, edges, and degree sequence
    Q = 10  # parameter from the paper ("On the uniform generation of random graphs with prescribed degree sequences")
    E = len(g.edges)
    edge_list = list(g.edges)

    r = nx.Graph()
    r.add_edges_from(edge_list)
    before_degree = dict(r.degree)

    if E == 1:
        return r

    for i in range(Q * E):
        edges = list(r.edges)
        if E < 2:
            print(edges)
        (a, b), (c, d) = random.sample(edges, 2)
        if len({a, b, c, d}) == 4:
            ## check if "a--d" and "c--b" are not connected in both directions
            if ((a, d) not in edges) & ((d, a) not in edges) & ((c, b) not in edges) & ((b, c) not in edges):
                edges.remove((a, b))
                edges.remove((c, d))
                edges.append((a, d))
                edges.append((c, b))

                #print("================== switch {num} ==================".format(num=i))
                #print((a,b), (c,d), '==>', (a,d), (c,b))
                #print("Edges: ", edges)
                ## if the switched network is not connected,
                ## try the other direction of switching.
                """
                if not nx.is_connected(r):
                    edges.remove((a, d))
                    edges.remove((c, b))
                    ## check if "a--c" and "d--b" are not connected in both directions
                    if ((a, c) not in edges) & ((c, a) not in edges) & ((d, b) not in edges) & ((b, d) not in edges):
                        edges.append((a, c))
                        edges.append((d, b))
                        r.clear()
                        r.add_edges_from(edges)
                        assert nx.is_connected(r)
                        print("Not connected, so try ", (a,b), (c,d), '==>', (a,c), (d,b))
                        print("Edges: ", edges)
                    else:
                        edges.append((a, b))
                        edges.append((c, d))
                """
    r.clear()
    r.add_edges_from(edges)
    after_degree = dict(r.degree)
    assert before_degree == after_degree
    return r


def generate_directed_random_networks_by_switching_algorithm(g):
    """
    parameter g: melodic network of a song
    """
    # generate a single random network by with a given number of nodes, edges, and degree sequence
    Q = 10  # parameter from the paper ("On the uniform generation of random graphs with prescribed degree sequences")
    E = len(g.edges)
    edge_list = list(g.edges)

    r = nx.DiGraph()
    r.add_edges_from(edge_list)
    before_degree = dict(r.degree)

    if E == 1:
        return r

    for i in range(Q * E):
        edges = list(r.edges)
        if E < 2:
            print(edges)
        (a, b), (c, d) = random.sample(edges, 2)
        if len({a, b, c, d}) == 4:
            if ((a, d) not in edges) & ((c, b) not in edges):
                edges.remove((a, b))
                edges.remove((c, d))
                edges.append((a, d))
                edges.append((c, b))

                #print("================== switch {num} ==================".format(num=i))
                #print((a, b), (c, d), '==>', (a, d), (c, b))
                #print("Edges: ", edges)
                """
                if not nx.is_weakly_connected(r):
                    ## if the switched network is not weakly connected,
                    ## go back to the original network before switching.
                    edges.append((a, b))
                    edges.append((c, d))
                    edges.remove((a, d))
                    edges.remove((c, b))
                    r.clear()
                    r.add_edges_from(edges)
                    assert nx.is_weakly_connected(r)
                    print("return to original")
                    print("Edges: ", edges)
                """
    r.clear()
    r.add_edges_from(edges)
    after_degree = dict(r.degree)
    assert before_degree == after_degree
    return r



"""
undirected_g = nx.Graph()
undirected_g.add_edges_from([(1,2),(2,3),(3,4),(4,5)])
generate_undirected_random_networks_by_switching_algorithm(undirected_g)

directed_g = nx.DiGraph()
directed_g.add_edges_from([(1,2),(2,3),(3,4),(4,5)])
generate_directed_random_networks_by_switching_algorithm(directed_g)

from helpers import load_data
dataInst = load_data()
test_g = nx.Graph()
test_g.add_edges_from(dataInst['Mozart1'].bigram())
test_directed_g = nx.DiGraph()
test_directed_g.add_edges_from(dataInst['Mozart1'].bigram())


small_nodes = []
big_nodes = []
adj_list = {}
for edge in map(str.split, nx.generate_adjlist(test_g)):
    this_node = int(edge[0])
    other_nodes = list(map(int, edge[1:]))
    for other_node in other_nodes:
        if this_node <= other_node:
            if this_node in adj_list:
                adj_list[this_node].append(other_node)
            else:
                adj_list[this_node] = [other_node]
            small_nodes.append(this_node)
            big_nodes.append(other_node)
        else:
            if other_node in adj_list:
                adj_list[other_node].append(this_node)
            else:
                adj_list[other_node] = [this_node]
            small_nodes.append(other_node)
            big_nodes.append(this_node)
"""


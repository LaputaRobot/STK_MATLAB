import collections
import copy
import math
import queue
import time

import matplotlib.pyplot as plt
from networkx import Graph
import networkx as nx
from numpy.random import default_rng

from PMetis.util import Common

MatchOrder = 'HE'
MatchScheme = 'EHEM'


def gen_init_graph(graph, link_loads):
    new_g = Graph(level=0)
    for node in graph:
        new_g.add_node('{}'.format(node[3:]), load=graph.nodes[node]['load'], real_load=0, contains=[], belong='',
                       inner_edge_wei=0)
    for edge in graph.edges:
        new_g.add_edge('{}'.format(edge[0][3:]), '{}'.format(edge[1][3:]), wei=link_loads[edge])
    return new_g


class Ctrl():
    def __init__(self):
        self.CoarsenTo = 240


def edge_equal(e1, e2):
    return e1[0] == e2[0] and e1[1] == e2[1]


def get_sum_wei_ex(graph: Graph, node, exclude_edge):
    sum = 0
    for edge in graph.edges(node):
        if not edge_equal(edge, exclude_edge):
            sum += graph.edges[edge]['wei']
    return sum


def get_match_node_load(graph: Graph, node1, node2):
    load = graph.nodes[node1]['load'] + graph.nodes[node2]['load']
    load += get_sum_wei_ex(graph, node1, (node1, node2))
    load += get_sum_wei_ex(graph, node2, (node2, node1))
    return load


def match_graph(graph: Graph):
    unmatched_nodes = {}
    for node in graph.nodes:
        unmatched_nodes[node] = get_node_order_val(graph, node)
    sorted_unmatched_nodes = dict(sorted(unmatched_nodes.items(), key=lambda x: x[1], reverse=True))
    index = 0
    while len(sorted_unmatched_nodes) > 0:
        node = list(sorted_unmatched_nodes.keys())[0]
        match_nei, max_wei = get_matched_neighbor(graph, sorted_unmatched_nodes, node)
        if match_nei is not None and max_wei > 0 and get_match_node_load(graph, node, match_nei) < 100:
            graph.nodes[match_nei]['belong'] = index
            sorted_unmatched_nodes.pop(match_nei)
        graph.nodes[node]['belong'] = index
        sorted_unmatched_nodes.pop(node)
        index += 1
    new_p = 0
    new_p_dict = {}
    for node in graph.nodes:
        p = graph.nodes[node]['belong']
        if p not in new_p_dict:
            new_p_dict[p] = new_p
            new_p += 1
        p = new_p_dict[p]
        graph.nodes[node]['belong'] = p
    return graph


def get_node_order_val(graph: Graph, node):
    neighbors = nx.neighbors(graph, node)
    node_val = 0
    if MatchOrder == 'HE':
        node_val = max([graph.edges[(node, nei)]['wei'] for nei in neighbors])
    if MatchOrder == 'EHEM':
        node_val = graph.nodes[node]['load']
        for nei in neighbors:
            node_val += (graph.nodes[nei]['load'] + graph.edges[(node, nei)]['wei'])
    return node_val


def get_matched_neighbor(graph: Graph, sorted_unmatched_nodes, node):
    neighbors = list(nx.neighbors(graph, node))
    match_nei = None
    max_wei = -math.inf
    if MatchScheme == 'EHEM':
        for nei in neighbors:
            if nei in sorted_unmatched_nodes:
                deepNeighbors = list(nx.neighbors(graph, nei))
                nei_wei = graph.edges[(node, nei)]['wei']
                for neiNei in deepNeighbors:
                    if neiNei != node:
                        nei_wei -= graph.edges[nei, neiNei]['wei']
                if match_nei is None or nei_wei > max_wei:
                    match_nei = nei
                    max_wei = nei_wei
    else:
        for nei in neighbors:
            if nei in sorted_unmatched_nodes:
                nei_wei = graph.nodes[nei]['load'] + graph.edges[(node, nei)]['wei']
                if match_nei is None or nei_wei > max_wei:
                    match_nei = nei
                    max_wei = nei_wei
    return match_nei, max_wei


def gen_coarse_graph(graph: Graph, level):
    coarse_g = Graph(level=level + 1)
    partitions = {}
    for node in graph:
        if graph.nodes[node]['belong'] in partitions:
            partitions[graph.nodes[node]['belong']].append(node)
        else:
            partitions[graph.nodes[node]['belong']] = [node]
    for p in partitions:
        inner_edge_wei = sum([graph.nodes[node]['inner_edge_wei'] for node in partitions[p]])
        if len(partitions[p]) > 1:
            inner_edge_wei += graph.edges[(partitions[p][0], partitions[p][1])]['wei']
        coarse_g.add_node('{}'.format(p), contains=partitions[p],
                          load=sum([graph.nodes[node]['load'] for node in partitions[p]]),
                          inner_edge_wei=inner_edge_wei)
    for p1 in partitions:
        for p2 in partitions:
            if p1 != p2 and not coarse_g.has_edge('{}'.format(p1), '{}'.format(p2)):
                partition_link_wei = get_partition_link_wei(graph, partitions[p1], partitions[p2])
                if partition_link_wei >= 0:
                    coarse_g.add_edge('{}'.format(p1), '{}'.format(p2), wei=partition_link_wei)
    graph.graph['coarser'] = coarse_g
    coarse_g.graph['finer'] = graph
    return coarse_g


def get_partition_link_wei(graph: Graph, from_part, to_part):
    partition_link_wei = -1
    for n1 in from_part:
        for n2 in to_part:
            if graph.has_edge(n1, n2):
                partition_link_wei = max(partition_link_wei, 0) + graph.edges[(n1, n2)]['wei']
    return partition_link_wei


def check_sum_load(graph):
    # print(graph.edges(data=True))
    edge_wei = 0
    for edge in graph.edges:
        edge_wei += graph.edges[edge]['wei']
    node_val = 0
    for node in graph.nodes:
        node_val += graph.nodes[node]['load']
    print('总节点：{}, 总节点负载：{}, 总链路负载：{}'.format(graph.number_of_nodes(), node_val, edge_wei))


def coarsen_graph(graph: Graph, ctrl: Ctrl):
    level = 0
    coarsen_to = ctrl.CoarsenTo
    while True:
        print('*' * 30, '第{}次粗化'.format(level), '*' * 30)
        check_sum_load(graph)
        match_graph(graph)
        level += 1
        coarse_g = gen_coarse_graph(graph, level)
        check_sum_load(coarse_g)
        graph = coarse_g
        if not (coarsen_to < graph.number_of_nodes() < 0.95 * graph.graph[
            'finer'].number_of_nodes() and graph.number_of_edges() > graph.number_of_nodes() / 2):
            break
    return graph


def init_KWay_partitioning(graph: Graph, part_num=8):
    pass


def recursive_part_graph(graph: Graph, part_num=8, iter_num=1):
    pass


def multilevel_bisect(graph: Graph, iter_num=1):
    for i in range(iter_num):
        cgraph = coarsen_graph(graph, coarsen_to=20)
        init_2way_partition(cgraph)
        refine_2way(graph, cgraph)


def refine_2way(graph: Graph, cgraph: Graph):
    pass


def init_2way_partition(graph: Graph, iter_num=1):
    sum_val = sum([graph.nodes[node]['load'] for node in graph.nodes])
    min_cut = math.inf
    for seed in range(iter_num):
        print('迭代{}'.format(seed))
        rng = default_rng(seed)
        start_node = rng.choice(list(graph.nodes))
        partition0 = set()
        tauched_node = set()
        partition1_sum_val = 0
        queue = collections.deque()
        queue.append(start_node)
        tauched_node.add(start_node)
        while len(queue) > 0 and partition1_sum_val < sum_val / 2:
            node = queue.popleft()
            partition0.add(node)
            graph.nodes[node]['belong'] = 0
            # print('add node {}'.format(node))
            partition1_sum_val += graph.nodes[node]['load']
            neighbors = nx.neighbors(graph, node)
            for nei in neighbors:
                if nei not in tauched_node:
                    tauched_node.add(nei)
                    queue.append(nei)
        for node in graph.nodes:
            if node not in partition0:
                graph.nodes[node]['belong'] = 1
        # 获取边界节点和切
        cut = 0
        boundary_nodes = [[], []]
        checked_node = set()
        node_gain = {}
        for node in graph.nodes:
            if node in checked_node:
                continue
            node_partition = graph.nodes[node]['belong']
            is_boundary_node = False
            checked_node.add(node)
            neighbors = nx.neighbors(graph, node)
            for nei in neighbors:
                nei_partition = graph.nodes[nei]['belong']
                if node_partition != nei_partition:
                    is_boundary_node = True
                    cut += graph.edges[(nei, node)]['wei']
                    # print('{}-x-{}: {}'.format(nei,node,graph.edges[(nei, node)]['wei']))
                    if nei not in checked_node:
                        boundary_nodes[nei_partition].append(nei)
                        checked_node.add(nei)
            if is_boundary_node:
                boundary_nodes[node_partition].append(node)
        FM_2WayRefine(graph)
        tmp_graph = copy.deepcopy(graph)
        for node in partition0:
            tmp_graph.remove_node(node)
        is_connected = nx.is_connected(tmp_graph)
        print('is_connected?: {}'.format(is_connected))
        if not is_connected:
            print([tmp_graph.subgraph(sub_graph).copy().nodes() for sub_graph in nx.connected_components(tmp_graph)])
        # if is_connected and cut<min_cut:
        if cut < min_cut:
            min_cut = cut
        print('partition1_sum_val: {}'.format(partition1_sum_val))
        print('partition1: {}'.format(partition0))
        print('partition1 len: {}'.format(len(partition0)))
        print('cut: {}'.format(cut))
        print('boundary_nodes: {}'.format(boundary_nodes))
    print(min_cut)


def FM_2WayRefine(graph: Graph, iter_num=5):
    pass


def run_metis_main(common: Common):
    origin_graph = gen_init_graph(common.graph, common.link_load)
    print(origin_graph)
    # print(len(coarse_graph_list))
    ctrl = Ctrl()
    coarsest_graph = coarsen_graph(origin_graph, ctrl)
    #ccds
    # print(coarsest_graph.edges(data=True))
    # print('最粗图节点数',coarsest_graph.number_of_nodes())
    # print(coarsest_graph.nodes(data=True))
    # print('二分最粗图')
    # multilevel_bisect(coarsest_graph)
    # plt.figure(dpi=200)
    # pos = nx.drawing.layout.spring_layout(coarsest_graph,seed=1)
    # labels = {n: '{}'.format(n.split('-')[1]) for n in pos}
    # nx.draw(coarsest_graph, pos,labels=labels, with_labels=True)
    # edge_weight = nx.get_edge_attributes(coarsest_graph, 'wei')
    # short_wei={}
    # for edge in edge_weight:
    #     short_wei[edge]='{:>3.2f}'.format(edge_weight[edge])
    # print(short_wei)
    # nx.draw_networkx_edge_labels(coarsest_graph, pos=pos, edge_labels=short_wei)
    # plt.show()

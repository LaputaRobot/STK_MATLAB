from networkx import Graph
import networkx as nx
from util import Common, Ctrl
from config import nparts, MatchOrder, MatchScheme
import math
import  numpy as np


def coarsen_graph(graph: Graph, ctrl: Ctrl):
    level = 0
    coarsen_to = ctrl.coarsenTo
    while True:
        # print( '第{}次粗化'.format(level),end=', ')
        # check_sum_load(graph)
        match_graph(graph, ctrl)
        level += 1
        coarse_g = gen_coarse_graph(graph, level)
        graph = coarse_g
        if not (coarsen_to < graph.number_of_nodes() < 0.95 * graph.graph[
            'finer'].number_of_nodes() and graph.number_of_edges() > graph.number_of_nodes() / 2):
            # check_sum_load(coarse_g)
            break
    return graph


def get_node_order_val(graph: Graph, node):
    '''
    获取节点匹配时的值, 该值越大, 在匹配时月在先被选择
    :param graph: 原始图
    :param node: 需要获取值的节点
    :return: 节点的值
    '''
    neighbors = nx.neighbors(graph, node)
    node_val = 0
    if MatchOrder == 'HE':
        node_val = max([graph.edges[(node, nei)]['wei'] for nei in neighbors])
    if MatchOrder == 'EHEM':
        node_val = graph.nodes[node]['load']
        for nei in neighbors:
            node_val += (graph.nodes[nei]['load'] + graph.edges[(node, nei)]['wei'])
    return node_val


def edge_equal(e1, e2):
    '''
    以元组形式判断两个边是否相等
    :param e1:
    :param e2:
    :return:
    '''
    return e1[0] == e2[0] and e1[1] == e2[1]


def get_sum_wei_ex(graph: Graph, node, exclude_edge):
    '''
    获取节点除了指定边后, 所有边的权重之和
    :param graph:
    :param node:
    :param exclude_edge:
    :return:
    '''
    sum = 0
    for edge in graph.edges(node):
        if not edge_equal(edge, exclude_edge):
            sum += graph.edges[edge]['wei']
    return sum


def get_match_node_load(graph: Graph, node1, node2):
    '''
    获取两个节点结合后的负载
    :param graph:
    :param node1:
    :param node2:
    :return:
    '''
    load = graph.nodes[node1]['load'] + graph.nodes[node2]['load']
    load += get_sum_wei_ex(graph, node1, (node1, node2))
    load += get_sum_wei_ex(graph, node2, (node2, node1))
    return load


def match_graph(graph: Graph, ctrl: Ctrl):
    unmatched_nodes = {}
    rng = np.random.default_rng(ctrl.seed)
    nodes=list(graph.nodes)
    rng.shuffle(nodes)
    for node in nodes:
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
    print('总节点：{}, 总节点负载：{}, 总链路负载：{:>7.2f}'.format(graph.number_of_nodes(), node_val, edge_wei))
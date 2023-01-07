from networkx import Graph
from util import Ctrl
import math
import copy
import networkx as nx
from coarsen import coarsen_graph
from numpy.random import default_rng
import collections


def part_graph_recursive(graph: Graph, ctrl: Ctrl):
    pass


def M_level_recursive_bisection(graph: Graph, ctrl: Ctrl, nParts, fPart):
    multilevel_bisect(graph, ctrl)


def multilevel_bisect(graph: Graph, ctrl: Ctrl):
    ctrl.nIparts = 5
    for i in range(ctrl.nCuts):
        cgraph = coarsen_graph(graph, ctrl)
        init_2way_partition(cgraph, ctrl)
        refine_2way(graph, cgraph)


def refine_2way(graph: Graph, cgraph: Graph):
    pass


def init_2way_partition(graph: Graph, ctrl: Ctrl):
    sum_val = sum([graph.nodes[node]['load'] for node in graph.nodes])
    min_cut = math.inf
    best_cut_graph = None
    unconnected_min_cut = math.inf
    seed = 0
    while seed < ctrl.nIparts or best_cut_graph is None:
        seed += 1
        # for seed in range(ctrl.nIparts):
        print('迭代{}'.format(seed))
        rng = default_rng(seed)
        start_node = rng.choice(list(graph.nodes))
        partition0 = set()
        touched_node = set()
        partition0_sum_val = 0
        queue = collections.deque()
        queue.append(start_node)
        touched_node.add(start_node)
        while len(queue) > 0 and partition0_sum_val < sum_val / 2:
            node = queue.popleft()
            partition0.add(node)
            graph.nodes[node]['belong'] = 0
            # print('add node {}'.format(node))
            partition0_sum_val += graph.nodes[node]['load']
            neighbors = nx.neighbors(graph, node)
            for nei in neighbors:
                if nei not in touched_node:
                    touched_node.add(nei)
                    queue.append(nei)
        for node in graph.nodes:
            if node not in partition0:
                graph.nodes[node]['belong'] = 1
        # 获取边界节点和切
        cut = 0
        boundary_nodes = [[], []]
        checked_node = set()
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
        tmp_graph.remove_nodes_from(partition0)
        is_connected = nx.is_connected(tmp_graph)
        print('is_connected?: {}'.format(is_connected))
        if not is_connected:
            if cut < unconnected_min_cut:
                unconnected_min_cut = cut
            print([tmp_graph.subgraph(sub_graph).copy().nodes() for sub_graph in nx.connected_components(tmp_graph)])
        if is_connected and cut < min_cut:
            # if cut < min_cut:
            min_cut = cut
            best_cut_graph = copy.deepcopy(graph)
        # print('partition0_sum_val: {}'.format(partition0_sum_val))
        # print('partition0: {}'.format(partition0))
        # print('partition0 len: {}'.format(len(partition0)))
        print('cut: {}'.format(cut))
        # print('boundary_nodes: {}'.format(boundary_nodes))
    if unconnected_min_cut < min_cut:
        print("unconnected {} better than {}, less {} ".format(unconnected_min_cut, min_cut,
                                                               min_cut - unconnected_min_cut))
    for node in graph:
        graph.nodes[node]['belong'] = best_cut_graph.nodes[node]['belong']
    print('min_cut: {} '.format(min_cut))


def FM_2WayRefine(graph: Graph, iter_num=5):
    pass

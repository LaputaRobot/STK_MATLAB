import queue
import random
import numpy as np
from networkx import Graph
from util import Ctrl
import math
import copy
import networkx as nx
from coarsen import coarsen_graph
from numpy.random import default_rng
import collections
import queue
import traceback
import matplotlib.pyplot as plt
from config import allow_err


def part_graph_recursive(graph: Graph, ctrl: Ctrl):
    pass


def M_level_recursive_bisection(graph: Graph, ctrl: Ctrl, nParts, fPart):
    multilevel_bisect(graph, ctrl)
    # TODO 拆分左右图
    # TODO 递归二分


def multilevel_bisect(graph: Graph, ctrl: Ctrl):
    ctrl.nIparts = 5
    for i in range(ctrl.nCuts):
        ctrl.seed = i
        cgraph = coarsen_graph(graph, ctrl)
        init_2way_partition(cgraph, ctrl)
        refine_2way(graph, cgraph)
    ctrl.seed = 0


def refine_2way(graph: Graph, cgraph: Graph):
    pass


def init_2way_partition(graph: Graph, ctrl: Ctrl):
    sum_val = sum([graph.nodes[node]['load'] for node in graph.nodes])
    min_cut = math.inf
    best_cut_graph = None
    unconnected_min_cut = math.inf
    seed = 0
    p0 = 0
    p1 = 0
    # nx.draw_networkx(graph)
    # plt.show()
    while seed < ctrl.nIparts or best_cut_graph is None:
        seed += 1
        # for seed in range(ctrl.nIparts):
        print('迭代{}'.format(seed), end=': \n')
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
        boundary_nodes = []
        node_ed = {}
        node_id = {}
        for node in graph.nodes:
            node_id[node] = 0
            node_ed[node] = 0
            node_partition = graph.nodes[node]['belong']
            neighbors = nx.neighbors(graph, node)
            for nei in neighbors:
                nei_partition = graph.nodes[nei]['belong']
                if node_partition != nei_partition:
                    cut += graph.edges[(node, nei)]['wei']
                    node_ed[node] += graph.edges[(node, nei)]['wei']
                else:
                    node_id[node] += graph.edges[(node, nei)]['wei']
            if node_ed[node] > 0 or len(graph[node]) == 0:
                boundary_nodes.append(node)
                # print("before add boundary node {}".format(node))

        # 优化二分
        print('{}: {}'.format(len(boundary_nodes), sorted(boundary_nodes)))
        where = {}
        for node in graph.nodes:
            where[node] = graph.nodes[node]['belong']
        new_cut, p_vs, swaps = FM_2WayRefine(graph, ctrl, boundary_nodes, cut, sum_val, partition0_sum_val, node_id,
                                             node_ed)
        s = []
        for node in graph.nodes:
            if where[node] != graph.nodes[node]['belong']:
                s.append(node)
        print('{}: {}'.format(len(boundary_nodes), sorted(boundary_nodes)))
        if new_cut == cut:
            print('------------same')
        # if set(s) != set(swaps):
        #     print('s: {}, swaps: {}'.format(s, swaps))
        assert set(s) == set(swaps)
        assert sum(p_vs) == sum_val
        print('new cut: {}, cut: {}, p_vs: {}->{}'.format(new_cut, cut,
                                                          [partition0_sum_val, sum_val - partition0_sum_val], p_vs))
        tmp_graph = copy.deepcopy(graph)
        tmp_graph.remove_nodes_from(partition0)
        is_connected = nx.is_connected(tmp_graph)
        # print('is_connected?: {}'.format(is_connected))
        if not is_connected:
            if cut < unconnected_min_cut:
                unconnected_min_cut = cut
            # print([tmp_graph.subgraph(sub_graph).copy().nodes() for sub_graph in nx.connected_components(tmp_graph)])
        if is_connected and cut < min_cut:
            # if cut < min_cut:
            min_cut = cut
            best_cut_graph = copy.deepcopy(graph)
            p0 = partition0_sum_val
            p1 = sum_val - partition0_sum_val
        # print('partition0_sum_val: {}, partition1_sum_val: {}'.format(partition0_sum_val, sum_val - partition0_sum_val))
        # print('partition0: {}'.format(partition0))
        # print('partition0 len: {}'.format(len(partition0)))
        print('cut: {:>7.2f}'.format(cut))
        # print('boundary_nodes: {}'.format(boundary_nodes))
    if unconnected_min_cut < min_cut:
        print("unconnected {:>7.2f} better {:>7.2f}, less {:>7.2f} ".format(unconnected_min_cut, min_cut,
                                                                            min_cut - unconnected_min_cut))
    for node in graph:
        graph.nodes[node]['belong'] = best_cut_graph.nodes[node]['belong']
    print('min_cut: {:>7.2f} p0: {}, p1: {}, unfactor: {:>7.2f}'.format(min_cut, p0, p1, max(p0, p1) / (sum_val / 2)))


def FM_2WayRefine(graph: Graph, ctrl: Ctrl, boundary_nodes, cut, sum_val, partition0_sum_val, node_id, node_ed):
    limit = min(max(0.01 * graph.number_of_nodes(), 15), 100)
    avgVwgt = min(sum_val / 20, 2 * sum_val / graph.number_of_nodes())
    queues = [queue.PriorityQueue(), queue.PriorityQueue()]
    p_vals = [partition0_sum_val, sum_val - partition0_sum_val]
    orig_diff = abs(sum_val / 2 - partition0_sum_val)
    moved = {}
    swaps = []
    source_id_ed = {}
    graph_cut = cut
    all_swaps = []
    for i in range(ctrl.niter):
        queues[0].queue.clear()
        queues[1].queue.clear()
        min_cut_order = -1
        new_cut = min_cut = init_cut = graph_cut
        min_diff = abs(sum_val / 2 - p_vals[0])
        np.random.seed(i)
        for node in np.random.choice(boundary_nodes, len(boundary_nodes), replace=False):
            p = graph.nodes[node]['belong']
            queues[p].put((node_id[node] - node_ed[node], node))
            source_id_ed[node] = node_id[node] - node_ed[node]
        nSwaps = 0
        swaps = []
        while nSwaps < graph.number_of_nodes():
            from_part = 0 if p_vals[0] > p_vals[1] else 1
            to_part = (from_part + 1) % 2
            if queues[from_part].empty():
                break
            high_gain_node = queues[from_part].get()[1]
            new_cut -= (node_ed[high_gain_node] - node_id[high_gain_node])
            p_vals[from_part] -= graph.nodes[high_gain_node]['load']
            p_vals[to_part] += graph.nodes[high_gain_node]['load']
            if ((new_cut < min_cut and abs(sum_val / 2 - p_vals[0]) <= orig_diff + avgVwgt) or
                    (new_cut == min_cut and abs(sum_val / 2 - p_vals[0]) < min_diff)):
                min_cut = new_cut
                min_diff = abs(sum_val / 2 - p_vals[0])
                min_cut_order = nSwaps
            elif nSwaps - min_cut_order > limit:
                new_cut += (node_ed[high_gain_node] - node_id[high_gain_node])
                p_vals[from_part] += graph.nodes[high_gain_node]['load']
                p_vals[to_part] -= graph.nodes[high_gain_node]['load']
                break
            # print('move {} from {} to {}'.format(high_gain_node, graph.nodes[high_gain_node]['belong'], to_part))
            graph.nodes[high_gain_node]['belong'] = to_part
            moved[high_gain_node] = nSwaps
            swaps.append(high_gain_node)
            # 更新id/ed信息
            node_id[high_gain_node], node_ed[high_gain_node] = node_ed[high_gain_node], node_id[high_gain_node]
            if abs(node_ed[high_gain_node]) <= allow_err and len(graph[high_gain_node]) > 0:
                boundary_nodes.remove(high_gain_node)
                # print("moving, remove boundary high_gain_node {}".format(high_gain_node))
            for nei in graph.neighbors(high_gain_node):
                nei_belong = graph.nodes[nei]['belong']
                symbol = 1 if nei_belong == graph.nodes[high_gain_node]['belong'] else -1
                node_id[nei] += graph.edges[(nei, high_gain_node)]['wei'] * symbol
                node_ed[nei] -= graph.edges[(nei, high_gain_node)]['wei'] * symbol
                if nei in boundary_nodes:
                    if abs(node_ed[nei]) <= allow_err:
                        boundary_nodes.remove(nei)
                        # print("moving remove boundary nei node {}".format(nei))
                        if nei not in moved:
                            queues[nei_belong].queue.remove((source_id_ed[nei], nei))
                    else:
                        if nei not in moved:
                            queues[nei_belong].queue.remove((source_id_ed[nei], nei))
                            queues[nei_belong].put((node_id[nei] - node_ed[nei], nei))
                            source_id_ed[nei] = node_id[nei] - node_ed[nei]
                else:
                    if node_ed[nei] > 0:
                        boundary_nodes.append(nei)
                        # print("moving add boundary node {}".format(nei))
                        if nei not in moved:
                            queues[nei_belong].put((node_id[nei] - node_ed[nei], nei))
                            source_id_ed[nei] = node_id[nei] - node_ed[nei]

            nSwaps += 1
        moved = {}
        nSwaps -= 1
        while nSwaps > min_cut_order:
            nSwaps -= 1
            high_gain_node = swaps.pop()
            now_belong = graph.nodes[high_gain_node]['belong']
            to_part = graph.nodes[high_gain_node]['belong'] = (now_belong + 1) % 2
            node_id[high_gain_node], node_ed[high_gain_node] = node_ed[high_gain_node], node_id[high_gain_node]
            if abs(node_ed[high_gain_node]) <= allow_err and high_gain_node in boundary_nodes and len(
                    graph[high_gain_node]) > 0:
                boundary_nodes.remove(high_gain_node)
                # print("restore remove boundary high_gain_node {}".format(high_gain_node))
            else:
                if node_ed[high_gain_node] > 0 and high_gain_node not in boundary_nodes:
                    boundary_nodes.append(high_gain_node)
                    # print("restore add boundary high_gain_node {}".format(high_gain_node))
            p_vals[now_belong] -= graph.nodes[high_gain_node]['load']
            p_vals[to_part] += graph.nodes[high_gain_node]['load']
            for nei in graph.neighbors(high_gain_node):
                nei_belong = graph.nodes[nei]['belong']
                symbol = 1 if to_part == nei_belong else -1
                node_id[nei] += graph.edges[(nei, high_gain_node)]['wei'] * symbol
                node_ed[nei] -= graph.edges[(nei, high_gain_node)]['wei'] * symbol
                if nei not in boundary_nodes and node_ed[nei] > 0:
                    boundary_nodes.append(nei)
                    # print("restore add boundary nei node {}".format(nei))
                if nei in boundary_nodes and abs(node_ed[nei]) <= allow_err:
                    boundary_nodes.remove(nei)
                    # print("restore remove boundary nei node {}".format(nei))

        graph_cut = min_cut
        for node in swaps:
            if node in all_swaps:
                all_swaps.remove(node)
            else:
                all_swaps.append(node)
        if min_cut_order <= 0 or min_cut == init_cut:
            print('iter {} break'.format(i))
            break
    return graph_cut, p_vals, all_swaps

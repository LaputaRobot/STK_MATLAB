import numpy as np
import networkx as nx
import heapq
import math

from networkx import Graph
from util import Ctrl, NodeGain
from config import *
from config import two_refine_log as log



def FM_2WayRefine(graph: Graph, ctrl: Ctrl):
    limit = min(max(0.01 * graph.number_of_nodes(), 15), 100)
    sum_val = graph.graph['sum_val']
    avgVwgt = min(sum_val / 20, 2 * sum_val / graph.number_of_nodes())
    queues = [[], []]
    orig_diff = abs(sum_val / 2 - graph.graph['p_vals'][0])
    moved = {}
    # source_id_ed = {}
    all_swaps = []
    for i in range(ctrl.niter):
        queues[0] = []
        queues[1] = []
        min_cut_order = -1
        new_cut = min_cut = init_cut = graph.graph['cut']
        min_diff = abs(sum_val / 2 - graph.graph['p_vals'][0])
        np.random.seed(i)
        for node in np.random.choice(graph.graph['boundary'], len(graph.graph['boundary']), replace=False):
            p = graph.nodes[node]['belong']
            queues[p].append(NodeGain(node, graph.graph['node_ed']
                             [node] - graph.graph['node_id'][node]))
            # source_id_ed[node] = graph.graph['node_id'][node] - graph.graph['node_ed'][node]
        for q in queues:
            heapq.heapify(q)
        nSwaps = 0
        swaps = []
        while nSwaps < graph.number_of_nodes():
            from_part = 0 if graph.graph['p_vals'][0] > graph.graph['p_vals'][1] else 1
            to_part = (from_part + 1) % 2
            if len(queues[from_part]) == 0:
                break
            heapq.heapify(queues[from_part])
            high_gain_node = heapq.heappop(queues[from_part]).node
            new_cut -= (graph.graph['node_ed'][high_gain_node] -
                        graph.graph['node_id'][high_gain_node])
            graph.graph['p_vals'][from_part] -= graph.nodes[high_gain_node]['load']
            graph.graph['p_vals'][to_part] += graph.nodes[high_gain_node]['load']
            if ((new_cut < min_cut and abs(sum_val / 2 - graph.graph['p_vals'][0]) <= orig_diff + avgVwgt) or
                    (new_cut == min_cut and abs(sum_val / 2 - graph.graph['p_vals'][0]) < min_diff)):
                min_cut = new_cut
                min_diff = abs(sum_val / 2 - graph.graph['p_vals'][0])
                min_cut_order = nSwaps
            elif nSwaps - min_cut_order > limit:
                new_cut += (graph.graph['node_ed'][high_gain_node] -
                            graph.graph['node_id'][high_gain_node])
                graph.graph['p_vals'][from_part] += graph.nodes[high_gain_node]['load']
                graph.graph['p_vals'][to_part] -= graph.nodes[high_gain_node]['load']
                break
            # print('move {} from {} to {}'.format(high_gain_node, graph.nodes[high_gain_node]['belong'], to_part))
            graph.nodes[high_gain_node]['belong'] = to_part
            moved[high_gain_node] = nSwaps
            swaps.append(high_gain_node)
            # 更新id/ed信息
            graph.graph['node_id'][high_gain_node], graph.graph['node_ed'][high_gain_node] = graph.graph['node_ed'][
                high_gain_node], graph.graph['node_id'][high_gain_node]
            if abs(graph.graph['node_ed'][high_gain_node]) <= allow_err and len(graph[high_gain_node]) > 0:
                graph.graph['boundary'].remove(high_gain_node)
                # print("moving, remove boundary high_gain_node {}".format(high_gain_node))
            for nei in graph.neighbors(high_gain_node):
                nei_belong = graph.nodes[nei]['belong']
                symbol = 1 if nei_belong == graph.nodes[high_gain_node]['belong'] else -1
                graph.graph['node_id'][nei] += graph.edges[(
                    nei, high_gain_node)]['wei'] * symbol
                graph.graph['node_ed'][nei] -= graph.edges[(
                    nei, high_gain_node)]['wei'] * symbol
                if nei in graph.graph['boundary']:
                    if abs(graph.graph['node_ed'][nei]) <= allow_err:
                        graph.graph['boundary'].remove(nei)
                        # print("moving remove boundary nei node {}".format(nei))
                        if nei not in moved:
                            queues[nei_belong].remove(NodeGain(nei, 0))
                    else:
                        if nei not in moved:
                            queues[nei_belong].remove(NodeGain(nei, 0))
                            queues[nei_belong].append(
                                NodeGain(nei, graph.graph['node_ed'][nei] - graph.graph['node_id'][nei]))
                            # source_id_ed[nei] = graph.graph['node_id'][nei] - graph.graph['node_ed'][nei]
                else:
                    if graph.graph['node_ed'][nei] > 0:
                        graph.graph['boundary'].append(nei)
                        # print("moving add boundary node {}".format(nei))
                        if nei not in moved:
                            queues[nei_belong].append(
                                NodeGain(nei, graph.graph['node_ed'][nei] - graph.graph['node_id'][nei]))
                            # source_id_ed[nei] = graph.graph['node_id'][nei] - graph.graph['node_ed'][nei]

            nSwaps += 1
        moved = {}
        nSwaps -= 1
        # log.info('swap: {}, min_cut_order: {}'.format(swaps, min_cut_order))
        while nSwaps > min_cut_order:
            nSwaps -= 1
            high_gain_node = swaps.pop()
            now_belong = graph.nodes[high_gain_node]['belong']
            to_part = graph.nodes[high_gain_node]['belong'] = (
                now_belong + 1) % 2
            graph.graph['node_id'][high_gain_node], graph.graph['node_ed'][high_gain_node] = graph.graph['node_ed'][
                high_gain_node], graph.graph['node_id'][high_gain_node]
            if abs(graph.graph['node_ed'][high_gain_node]) <= allow_err and high_gain_node in graph.graph[
                    'boundary'] and len(
                    graph[high_gain_node]) > 0:
                graph.graph['boundary'].remove(high_gain_node)
                # print("restore remove boundary high_gain_node {}".format(high_gain_node))
            else:
                if graph.graph['node_ed'][high_gain_node] > 0 and high_gain_node not in graph.graph['boundary']:
                    graph.graph['boundary'].append(high_gain_node)
                    # print("restore add boundary high_gain_node {}".format(high_gain_node))
            graph.graph['p_vals'][now_belong] -= graph.nodes[high_gain_node]['load']
            graph.graph['p_vals'][to_part] += graph.nodes[high_gain_node]['load']
            for nei in graph.neighbors(high_gain_node):
                nei_belong = graph.nodes[nei]['belong']
                symbol = 1 if to_part == nei_belong else -1
                graph.graph['node_id'][nei] += graph.edges[(
                    nei, high_gain_node)]['wei'] * symbol
                graph.graph['node_ed'][nei] -= graph.edges[(
                    nei, high_gain_node)]['wei'] * symbol
                if nei not in graph.graph['boundary'] and graph.graph['node_ed'][nei] > 0:
                    graph.graph['boundary'].append(nei)
                    # print("restore add boundary nei node {}".format(nei))
                if nei in graph.graph['boundary'] and abs(graph.graph['node_ed'][nei]) <= allow_err:
                    graph.graph['boundary'].remove(nei)
                    # print("restore remove boundary nei node {}".format(nei))

        # log.info('swap: {}'.format(swaps))
        graph.graph['cut'] = min_cut
        for node in swaps:
            if node in all_swaps:
                all_swaps.remove(node)
            else:
                all_swaps.append(node)

        if min_cut_order <= 0 or min_cut == init_cut:
            # log.info('iter {} break by min_cut_order <= 0? {}, min_cut == init_cut? {}'.format(i,min_cut_order <= 0,min_cut == init_cut))
            break
    return all_swaps


def refine_2way(ograph: Graph, cgraph: Graph, ctrl: Ctrl):
    src_cut = cgraph.graph['cut']
    while True:
        compute_2way_partition_params(cgraph)
        # print('before {} cut: {}'.format(cgraph.number_of_nodes(),cgraph.graph['cut']))
        FM_2WayRefine(cgraph, ctrl)
        # print('after {} cut: {}'.format(cgraph.number_of_nodes(),cgraph.graph['cut']))
        if ograph == cgraph:
            break
        project_2Way_partition(ctrl, cgraph)
        # print("投影: {} -> {}".format(cgraph.number_of_nodes(),cgraph.graph['finer'].number_of_nodes()))
        cgraph = cgraph.graph['finer']
    refine_gain = src_cut-cgraph.graph['cut']
    log.info('finish refine_2way,\t cut: {:>7.2f}, p_vals: {}, unfactor: {:>7.2f}, refine_gain: {:>7.2f}'.format(
        cgraph.graph['cut'], cgraph.graph['p_vals'], max(
            cgraph.graph['p_vals'])/(sum(cgraph.graph['p_vals'])/2),
        refine_gain))


def compute_2way_partition_params(graph: Graph):
    """优化二分时所需参数， 如边界节点、节点的ed/id, 图的切

    Args:
        graph (Graph): 待优化的图
    """
    cut = 0
    boundary_nodes = []
    node_ed = {}
    node_id = {}
    partition0_sum_val = 0
    sum_val = 0
    for node in graph.nodes:
        node_id[node] = 0
        node_ed[node] = 0
        node_partition = graph.nodes[node]['belong']
        sum_val += graph.nodes[node]['load']
        if node_partition == 0:
            partition0_sum_val += graph.nodes[node]['load']
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
    graph.graph['sum_val'] = sum_val
    graph.graph['boundary'] = boundary_nodes
    graph.graph['cut'] = cut / 2
    graph.graph['p_vals'] = [partition0_sum_val,
                             graph.graph['sum_val'] - partition0_sum_val]
    graph.graph['node_id'] = node_id
    graph.graph['node_ed'] = node_ed


def compute_load_imbalance(graph: Graph, n):
    p_vals = graph.graph['p_vals']
    max_p_vals = max(p_vals)
    sum_val = sum(p_vals)
    tar_p_val = sum_val/n
    pij = 1/sum_val/(1/n)
    ub_vec = pow(un_factor, 1/math.log(nparts))
    return max_p_vals*pij-ub_vec

def project_2Way_partition(ctrl: Ctrl, cgraph: Graph):
    """将图投影到更细化的图

    Args:
        ctrl (Ctrl): 控制参数
        cgraph (Graph): 粗图
    """
    graph = cgraph.graph['finer']
    for node in cgraph:
        contains = cgraph.nodes[node]['contains']
        for con_node in contains:
            graph.nodes[con_node]['belong'] = cgraph.nodes[node]['belong']

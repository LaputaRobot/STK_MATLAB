import queue
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
from config import allow_err


def part_graph_recursive(graph: Graph, ctrl: Ctrl):
    pass


def M_level_recursive_bisection(graph: Graph, ctrl: Ctrl, nParts, fPart):
    cut = multilevel_bisect(graph, ctrl)
    # TODO 拆分左右图
    # TODO 递归二分


def multilevel_bisect(graph: Graph, ctrl: Ctrl):
    ctrl.nIparts = 5
    min_cut = math.inf
    best_bal = math.inf
    best_graph = None
    for i in range(ctrl.nCuts):
        ctrl.seed = i
        cgraph = coarsen_graph(graph, ctrl)
        init_2way_partition(cgraph, ctrl)
        # src_cut = cgraph.graph['cut']
        # print(' 初始划分后 cut: {}'.format(cgraph.graph['cut']))
        refine_2way(graph, cgraph, ctrl)
        print('细化后 cut: {}'.format(graph.graph['cut']))
        # if abs(src_cut-graph.graph['cut']) > allow_err:
        #     print(
        #         "有效----------------------------------------    {}".format(src_cut-graph.graph['cut']))
        if i == 0 or (get_graph_unfactor(graph, 2) < 1.6 and graph.graph['cut'] < min_cut) or (
                best_bal > 1.6 and get_graph_unfactor(graph, 2) < best_bal):
            best_bal = get_graph_unfactor(graph, 2)
            min_cut = graph.graph['cut']
            best_graph = copy.deepcopy(graph)
    copy_graph_info(best_graph, graph)
    ctrl.seed = 0
    return min_cut


def get_graph_unfactor(graph: Graph, part: int):
    return max(graph.graph['p_vals'])/(sum(graph.graph['p_vals'])/part)


def refine_2way(ograph: Graph, cgraph: Graph, ctrl: Ctrl):
    # pass
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


def init_2way_partition(graph: Graph, ctrl: Ctrl):
    sum_val = sum([graph.nodes[node]['load'] for node in graph.nodes])
    graph.graph['sum_val'] = sum_val
    min_cut = math.inf
    best_cut_graph = None
    unconnected_min_cut = math.inf
    seed = 0
    # nx.draw_networkx(graph)
    # plt.show()
    while seed < ctrl.nIparts or best_cut_graph is None:
        seed += 1
        # for seed in range(ctrl.nIparts):
        # print('迭代{}'.format(seed), end=': \n')
        rng = default_rng(seed)
        start_node = rng.choice(list(graph.nodes))
        partition0 = set()
        touched_node = set()
        partition0_sum_val = 0
        queue = collections.deque()
        queue.append(start_node)
        touched_node.add(start_node)
        # 扩张图
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
        # print('{}: {}'.format(len(boundary_nodes), sorted(boundary_nodes)))
        where = {}
        for node in graph.nodes:
            where[node] = graph.nodes[node]['belong']
        # 优化扩张后的二分图
        compute_2way_partition_params(graph)
        src_cut = graph.graph['cut']
        swaps = FM_2WayRefine(graph, ctrl)
        # 验证
        src_partition0 = partition0.copy()
        my_s = partition0.copy()
        partition0.clear()
        s = []
        for node in graph.nodes:
            if graph.nodes[node]['belong'] == 0:
                partition0.add(node)
            if where[node] != graph.nodes[node]['belong']:
                s.append(node)
        for node in swaps:
            if node in src_partition0:
                my_s.remove(node)
            else:
                my_s.add(node)
        assert my_s == partition0
        assert set(s) == set(swaps)

        # print('new cut: {:>7.2f}, src cut: {:>7.2f}, p_vs: {}->{}'.format(graph.graph['cut'], src_cut,
        #                                                                   [partition0_sum_val,
        #                                                                    sum_val - partition0_sum_val],
        #                                                                   graph.graph['p_vals']))
        cut = graph.graph['cut']
        tmp_graph = copy.deepcopy(graph)
        tmp_graph.remove_nodes_from(partition0)
        is_connected = nx.is_connected(tmp_graph)
        # print('is_connected?: {}'.format(is_connected))
        if not is_connected:
            if cut < unconnected_min_cut:
                unconnected_min_cut = cut
            # print([tmp_graph.subgraph(sub_graph).copy().nodes()
            #       for sub_graph in nx.connected_components(tmp_graph)])
        if is_connected and cut < min_cut:
            # if cut < min_cut:
            min_cut = cut
            best_cut_graph = copy.deepcopy(graph)
        # print('partition0_sum_val: {}, partition1_sum_val: {}'.format(partition0_sum_val, sum_val - partition0_sum_val))
        # print('partition0: {}'.format(partition0))
        # print('partition0 len: {}'.format(len(partition0)))
        # print('cut: {:>7.2f}'.format(cut))
        # print('boundary_nodes: {}'.format(boundary_nodes))
    if unconnected_min_cut < min_cut:
        pass
        # print("unconnected {:>7.2f} better {:>7.2f}, less {:>7.2f} ".format(unconnected_min_cut, min_cut,
        #                                                                     min_cut - unconnected_min_cut))
    copy_graph_info(best_cut_graph, graph)
    print('min_cut: {:>7.2f} p0: {}, p1: {}, unfactor: {:>7.2f}'.format(
        graph.graph['cut'], graph.graph['p_vals'][0], graph.graph['p_vals'][1], max(graph.graph['p_vals'][0], graph.graph['p_vals'][1]) / (sum_val / 2)))


def copy_graph_info(from_graph: Graph, to_graph: Graph):
    for node in to_graph:
        to_graph.nodes[node]['belong'] = from_graph.nodes[node]['belong']
    to_graph.graph['cut'] = from_graph.graph['cut']
    to_graph.graph['p_vals'] = from_graph.graph['p_vals']
    to_graph.graph['boundary'] = from_graph.graph['boundary']
    to_graph.graph['node_id'] = from_graph.graph['node_id']
    to_graph.graph['node_ed'] = from_graph.graph['node_ed']


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
    sum_val = sum([graph.nodes[node]['load'] for node in graph.nodes])
    graph.graph['sum_val'] = sum_val
    for node in graph.nodes:
        node_id[node] = 0
        node_ed[node] = 0
        node_partition = graph.nodes[node]['belong']
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
    graph.graph['boundary'] = boundary_nodes
    graph.graph['cut'] = cut / 2
    graph.graph['p_vals'] = [partition0_sum_val,
                             graph.graph['sum_val'] - partition0_sum_val]
    graph.graph['node_id'] = node_id
    graph.graph['node_ed'] = node_ed


def FM_2WayRefine(graph: Graph, ctrl: Ctrl):
    limit = min(max(0.01 * graph.number_of_nodes(), 15), 100)
    sum_val = graph.graph['sum_val']
    avgVwgt = min(sum_val / 20, 2 * sum_val / graph.number_of_nodes())
    queues = [queue.PriorityQueue(), queue.PriorityQueue()]
    orig_diff = abs(sum_val / 2 - graph.graph['p_vals'][0])
    moved = {}
    source_id_ed = {}
    all_swaps = []
    for i in range(ctrl.niter):
        queues[0].queue.clear()
        queues[1].queue.clear()
        min_cut_order = -1
        new_cut = min_cut = init_cut = graph.graph['cut']
        min_diff = abs(sum_val / 2 - graph.graph['p_vals'][0])
        np.random.seed(i)
        for node in np.random.choice(graph.graph['boundary'], len(graph.graph['boundary']), replace=False):
            p = graph.nodes[node]['belong']
            queues[p].put((graph.graph['node_id'][node] -
                          graph.graph['node_ed'][node], node))
            source_id_ed[node] = graph.graph['node_id'][node] - \
                graph.graph['node_ed'][node]
        nSwaps = 0
        swaps = []
        while nSwaps < graph.number_of_nodes():
            from_part = 0 if graph.graph['p_vals'][0] > graph.graph['p_vals'][1] else 1
            to_part = (from_part + 1) % 2
            if queues[from_part].empty():
                break
            high_gain_node = queues[from_part].get()[1]
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
                            queues[nei_belong].queue.remove(
                                (source_id_ed[nei], nei))
                    else:
                        if nei not in moved:
                            queues[nei_belong].queue.remove(
                                (source_id_ed[nei], nei))
                            queues[nei_belong].put(
                                (graph.graph['node_id'][nei] - graph.graph['node_ed'][nei], nei))
                            source_id_ed[nei] = graph.graph['node_id'][nei] - \
                                graph.graph['node_ed'][nei]
                else:
                    if graph.graph['node_ed'][nei] > 0:
                        graph.graph['boundary'].append(nei)
                        # print("moving add boundary node {}".format(nei))
                        if nei not in moved:
                            queues[nei_belong].put(
                                (graph.graph['node_id'][nei] - graph.graph['node_ed'][nei], nei))
                            source_id_ed[nei] = graph.graph['node_id'][nei] - \
                                graph.graph['node_ed'][nei]

            nSwaps += 1
        moved = {}
        nSwaps -= 1
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

        graph.graph['cut'] = min_cut
        for node in swaps:
            if node in all_swaps:
                all_swaps.remove(node)
            else:
                all_swaps.append(node)
        if min_cut_order <= 0 or min_cut == init_cut:
            # print('iter {} break'.format(i))
            break
    return all_swaps

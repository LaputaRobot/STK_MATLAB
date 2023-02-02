import queue
import collections
import math
import copy
import networkx as nx

from networkx import Graph
from util import Ctrl
from coarsen import coarsen_graph
from numpy.random import default_rng
from config import *
from k_refine import *
from b_refine import *
from config import init_log as log
from pprint import pprint



def M_level_recur_bisect(o_graph: Graph, graph: Graph, ctrl: Ctrl, nParts, fPart):
    log.info('# {}, nParts: {:>2.0f}, sum_val: {}'.format(
        graph.number_of_nodes(), nParts, graph.graph['sum_val']))
    # log.info('nodes: {}'.format(list(graph.nodes)))

    cut = multilevel_bisect(graph, ctrl)

    # print('二分: {}'.format(graph.number_of_nodes()))
    for node in graph:
        o_graph.nodes[node]['belong'] = graph.nodes[node]['belong']+fPart
    #  拆分左右图
    l_graph, r_graph = split_graph(graph)
    log.info('split to ')
    log.info('L graph: #{:3d}, val: {}'.format(
        l_graph.number_of_nodes(), l_graph.graph['sum_val']))
    # log.info('l nodes: {}'.format(list(l_graph.nodes)))
    log.info('R graph: #{:3d}, val: {}\n\n'.format(
        r_graph.number_of_nodes(), r_graph.graph['sum_val']))
    # log.info('r nodes: {}'.format(list(l_graph.nodes)))

    # print('split to {} {}'.format(sum([l_graph.nodes[node]['load'] for node in l_graph ]),sum([r_graph.nodes[node]['load'] for node in r_graph ])))
    # print('split to {} {}'.format(sum([l_graph.nodes[node]['load'] for node in l_graph ]),sum([r_graph.nodes[node]['load'] for node in r_graph ])))
    # 递归二分
    # print('is con? l:{}, r: {}'.format(nx.is_connected(l_graph),nx.is_connected(r_graph)))
    # print('num of nodes? l: {}, r: {}'.format(l_graph.number_of_nodes(),r_graph.number_of_nodes()))
    if nParts > 3:
        cut += M_level_recur_bisect(o_graph, l_graph, ctrl, nParts/2, fPart)
        cut += M_level_recur_bisect(o_graph, r_graph,
                                    ctrl, nParts - nParts/2, fPart+nParts/2)
    elif nParts == 3:
        cut += M_level_recur_bisect(o_graph,
                                    r_graph, ctrl, nParts-nParts/2, fPart+nParts/2)
    return cut


def split_graph(graph: Graph):
    l_nodes = []
    for node in graph:
        if graph.nodes[node]['belong'] == 0:
            l_nodes.append(node)
    l_graph = copy.deepcopy(graph)
    l_graph.remove_nodes_from(
        list(filter(lambda node: node not in l_nodes, graph.nodes)))
    r_graph = copy.deepcopy(graph)
    r_graph.remove_nodes_from(l_nodes)
    l_graph.graph['sum_val'] = sum(
        [l_graph.nodes[node]['load'] for node in l_graph])
    r_graph.graph['sum_val'] = graph.graph['sum_val']-l_graph.graph['sum_val']
    return l_graph, r_graph


def multilevel_bisect(graph: Graph, ctrl: Ctrl):
    ctrl.nIparts = 5
    min_cut = math.inf
    best_bal = math.inf
    best_graph = None
    log.info('+'*25+'-> start # {} multilevel bisect'.format(graph.number_of_nodes()))
    for i in range(ctrl.nCuts):
        ctrl.seed = i
        log.info('-'*10+'迭代------> {}'.format(i))
        cgraph = coarsen_graph(graph, ctrl)
        init_2way_partition(cgraph, ctrl)
        # src_cut = cgraph.graph['cut']
        # print('初始划分后 cut: {}'.format(cgraph.graph['cut']),end=', ')
        # l,r=split_graph(cgraph)
        # print('is con? {} {}'.format(nx.is_connected(l),nx.is_connected(r)))
        refine_2way(graph, cgraph, ctrl)

        # l,r=split_graph(graph)
        # print('is con? {} {}'.format(nx.is_connected(l),nx.is_connected(r)))
        # if abs(src_cut-graph.graph['cut']) > allow_err:
        #     log.info(
        #         "有效----------------------------------------    {}".format(src_cut-graph.graph['cut']))
        cur_bal = compute_load_imbalance(graph, 2, ctrl)
        log.info('bal: {:>7.4f}\n'.format(cur_bal))
        if i == 0 or (cur_bal < max_allow_bal and graph.graph['cut'] < min_cut) or (
                best_bal > max_allow_bal and cur_bal < best_bal):
            best_bal = cur_bal
            min_cut = graph.graph['cut']
            best_graph = copy.deepcopy(graph)
    copy_graph_info(best_graph, graph)
    ctrl.seed = 0
    log.info('+'*25+'->finish # {} bisect:\t cut: {:>7.2f}, p_vals: {}, bal: {:>7.4f}\n\n'.format(graph.number_of_nodes(),
                                                                                                  graph.graph['cut'], graph.graph['p_vals'], best_bal))
    return min_cut


def get_graph_unfactor(graph: Graph, part: int):
    return max(graph.graph['p_vals'])/((sum(graph.graph['p_vals'])/part))


def init_2way_partition(graph: Graph, ctrl: Ctrl):
    sum_val = sum([graph.nodes[node]['load'] for node in graph.nodes])
    graph.graph['sum_val'] = sum_val
    min_cut = math.inf
    best_cut_graph = None
    unconnected_min_cut = math.inf
    seed = 0

    min_p_wei = sum_val/2*ctrl.ubfactors
    max_p_wei = 1/ctrl.ubfactors*sum_val/2
    # nx.draw_networkx(graph)
    # plt.show()
    while seed < ctrl.nIparts or best_cut_graph is None:
        seed += 1
        # for seed in range(ctrl.nIparts):
        # print('初始二分迭代{}'.format(seed), end=': \n')
        rng = default_rng(seed+ctrl.seed*ctrl.nIparts)
        start_node = rng.choice(list(graph.nodes))
        partition0 = []
        touched_node = []
        untouched_node = list(graph.nodes)
        p0_val = 0
        queue = collections.deque()
        queue.append(start_node)
        touched_node.append(start_node)
        untouched_node.remove(start_node)
        drain = False
        # log.info('{}, {}'.format(seed+ctrl.seed*ctrl.nIparts, start_node))
        # 扩张图
        while True:
            if len(queue) == 0:
                if len(untouched_node) == 0 or drain:
                    break
                start_node = rng.choice(untouched_node)
                queue.append(start_node)
                touched_node.append(start_node)
                untouched_node.remove(start_node)
            node = queue.popleft()
            if p0_val > 0 and sum_val-p0_val-graph.nodes[node]['load'] < min_p_wei:
                drain = True
                continue
            partition0.append(node)
            graph.nodes[node]['belong'] = 0
            p0_val += graph.nodes[node]['load']
            if sum_val-p0_val <= max_p_wei:
                break

            drain = False
            neighbors = nx.neighbors(graph, node)
            for nei in neighbors:
                if nei not in touched_node:
                    touched_node.append(nei)
                    untouched_node.remove(nei)
                    queue.append(nei)
        for node in graph.nodes:
            if node not in partition0:
                graph.nodes[node]['belong'] = 1
        # print(p0_val,end=', ')
        # print('{}: {}'.format(len(boundary_nodes), sorted(boundary_nodes)))
        where = {}
        for node in graph.nodes:
            where[node] = graph.nodes[node]['belong']
        # 优化扩张后的二分图
        # print('优化前 {}'.format(nx.is_connected(graph.subgraph(partition0))))
        compute_2way_partition_params(graph)
        log.debug('p0: {}'.format(partition0))
        src_cut = graph.graph['cut']
        swaps = FM_2WayRefine(graph, ctrl)
        # 验证
        src_partition0 = partition0.copy()
        my_s = partition0.copy()
        partition0.clear()
        s = []
        for node in graph.nodes:
            if graph.nodes[node]['belong'] == 0:
                partition0.append(node)
            if where[node] != graph.nodes[node]['belong']:
                s.append(node)
        for node in swaps:
            if node in src_partition0:
                my_s.remove(node)
            else:
                my_s.append(node)
        assert set(my_s) == set(partition0)
        assert set(s) == set(swaps)

        log.debug('[{}], cut: {:>7.2f} -FL-> {:>5.2f}, p_vals: {} -FL-> {}'.format(seed, src_cut, graph.graph['cut'],
                                                                                  [p0_val,
                                                                                   sum_val - p0_val],
                                                                                  graph.graph['p_vals']))
        cut = graph.graph['cut']
        # 判断是否要连续
        # tmp_graph = copy.deepcopy(graph)
        # tmp_graph.remove_nodes_from(partition0)
        # p1_graph=graph.subgraph(partition0)
        # is_connected = nx.is_connected(tmp_graph)
        # print('is_connected?: {} {}'.format(nx.is_connected(p1_graph),is_connected))
        # if not is_connected:
        #     if cut < unconnected_min_cut:
        #         unconnected_min_cut = cut
        # print([tmp_graph.subgraph(sub_graph).copy().nodes()
        #       for sub_graph in nx.connected_components(tmp_graph)])
        # if is_connected and cut < min_cut:
        if cut < min_cut:
            min_cut = cut
            best_cut_graph = copy.deepcopy(graph)
        #     log.info('------------->')
        # else:
        #     log.info()
        # print('partition0_sum_val: {}, partition1_sum_val: {}'.format(partition0_sum_val, sum_val - partition0_sum_val))
        # print('partition0: {}'.format(partition0))
        # print('partition0 len: {}'.format(len(partition0)))
        # print('cut: {:>7.2f}'.format(cut))
        # print('boundary_nodes: {}'.format(boundary_nodes))
    # if unconnected_min_cut < min_cut:
    #     pass
        # print("unconnected {:>7.2f} better {:>7.2f}, less {:>7.2f} ".format(unconnected_min_cut, min_cut,
        #                                                                     min_cut - unconnected_min_cut))
    copy_graph_info(best_cut_graph, graph)
    log.info('finish init 2 part,\t cut: {:>7.2f}, p_vals: {}, unfactor: {:>7.2f}'.format(
        graph.graph['cut'], graph.graph['p_vals'], max(graph.graph['p_vals']) / (sum_val / 2)))


def get_node_part(graph: Graph):
    node_part = {}
    part_val = {}
    for node in graph:
        part = '{:>2.0f}'.format(graph.nodes[node]['belong'])
        if part in node_part:
            node_part[part].append(node)
            part_val[part] += graph.nodes[node]['load']
        else:
            node_part[part] = [node]
            part_val[part] = graph.nodes[node]['load']
    return node_part, part_val


def copy_graph_info(from_graph: Graph, to_graph: Graph):
    for node in to_graph:
        to_graph.nodes[node]['belong'] = from_graph.nodes[node]['belong']
    to_graph.graph['cut'] = from_graph.graph['cut']
    to_graph.graph['p_vals'] = from_graph.graph['p_vals']
    to_graph.graph['boundary'] = from_graph.graph['boundary']
    to_graph.graph['node_id'] = from_graph.graph['node_id']
    to_graph.graph['node_ed'] = from_graph.graph['node_ed']

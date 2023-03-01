import queue
import numpy as np
import heapq

from pprint import pprint,pformat
from config import *
from networkx import Graph
from util import *
from contig import *
from config import k_refine_log as log
from b_refine import compute_load_imbalance



def compute_k_way_params(graph: Graph):
    """计算K路划分参数, 如边界节点及节点ed/id

    Args:
        graph (Graph): 初始划分完成后的图
    """

    part = {}
    part_val = {}
    cut = 0
    boundary_nodes = []
    node_ed = {}
    node_id = {}
    node_kr_info = {}
    sum_val = 0
    for node in graph.nodes:
        node_p = graph.nodes[node]['belong']
        # node_p = '{:>2.0f}'.format(node_p)
        if node_p in part:
            part[node_p].append(node)
            part_val[node_p] += graph.nodes[node]['load']
        else:
            part[node_p] = [node]
            part_val[node_p] = graph.nodes[node]['load']
        node_id[node] = 0
        node_ed[node] = 0
        node_kr_info[node] = {}
        sum_val += graph.nodes[node]['load']
        neighbors = nx.neighbors(graph, node)
        for nei in neighbors:
            nei_partition = graph.nodes[nei]['belong']
            if node_p != nei_partition:
                cut += graph.edges[(node, nei)]['wei']
                node_ed[node] += graph.edges[(node, nei)]['wei']
            else:
                node_id[node] += graph.edges[(node, nei)]['wei']
        if node_ed[node] > 0:
            for nei in neighbors:
                nei_partition = graph.nodes[nei]['belong']
                if node_p != nei_partition:
                    if nei_partition in node_kr_info[node]:
                        node_kr_info[node][nei_partition] += graph.edges[(
                            node, nei)]['wei']
                    else:
                        node_kr_info[node][nei_partition] = graph.edges[(
                            node, nei)]['wei']
        if node_ed[node] - node_id[node] > 0:
            boundary_nodes.append(node)
            # print("before add boundary node {}".format(node))
    graph.graph['sum_val'] = sum_val
    graph.graph['boundary'] = boundary_nodes
    graph.graph['cut'] = cut / 2
    graph.graph['p_vals'] = part_val
    graph.graph['node_id'] = node_id
    graph.graph['node_ed'] = node_ed
    graph.graph['part'] = part
    graph.graph['node_kr_info'] = node_kr_info
    return part, part_val


def project_k_way_partition(ctrl: Ctrl, graph: Graph):
    c_graph = graph.graph['coarser']
    for node in c_graph:
        contains = c_graph.nodes[node]['contains']
        for con_node in contains:
            graph.nodes[con_node]['belong'] = c_graph.nodes[node]['belong']


def refine_k_way(graph: Graph, o_graph: Graph, ctrl: Ctrl):
    num_comps = find_components(graph)
    contiguous = ctrl.contiguous
    n_level = graph.graph['level']
    if contiguous and num_comps > ctrl.nparts:
        eliminate_components(graph, ctrl)
        compute_k_way_boundary(graph, ctrl, BALANCE)
        greedy_k_way_opt(graph, ctrl, 5, 0, BALANCE)
        compute_k_way_boundary(graph, ctrl, REFINE)
        greedy_k_way_opt(graph, ctrl, ctrl.niter, 0, REFINE)
        ctrl.contiguous = False
    i = 0
    while True:
        if 2*i >= n_level and not is_balanced(ctrl, graph, 0.02):
            compute_k_way_boundary(graph, ctrl, BALANCE)
            greedy_k_way_opt(graph, ctrl, 1, 0, BALANCE)
            compute_k_way_boundary(graph, ctrl, REFINE)
        greedy_k_way_opt(graph, ctrl, ctrl.niter, 5, REFINE)
        if contiguous and i == n_level / 2:
            if find_components(graph) > ctrl.nparts:
                eliminate_components(graph, ctrl)
            if not is_balanced(ctrl, graph, 0.02):
                ctrl.contiguous = 1
                compute_k_way_boundary(graph, ctrl, BALANCE)
                greedy_k_way_opt(graph, ctrl, 5, 0, BALANCE)
                compute_k_way_boundary(graph, ctrl, REFINE)
                greedy_k_way_opt(graph, ctrl, ctrl.niter, 0, REFINE)
                ctrl.contiguous = 0
        i += 1
        if graph == o_graph:
            break
        graph = graph.graph['finer']
        project_k_way_partition(ctrl, graph)
        compute_k_way_params(graph)
    ctrl.contiguous = contiguous
    if contiguous and find_components(graph) > ctrl.nparts:
        eliminate_components(graph, ctrl)
    if not is_balanced(ctrl, graph, 0):
        compute_k_way_boundary(graph, ctrl, BALANCE)
        greedy_k_way_opt(graph, ctrl, 5, 0, BALANCE)
        compute_k_way_boundary(graph, ctrl, REFINE)
        greedy_k_way_opt(graph, ctrl, ctrl.niter, 0, REFINE)


def is_balanced(ctrl: Ctrl, graph: Graph, f_factor):
    return compute_load_imbalance(graph, ctrl.nparts, ctrl) <= f_factor



def compute_k_way_boundary(graph: Graph, ctrl: Ctrl, b_type):
    boundary_nodes = []
    node_ed = graph.graph['node_ed']
    node_id = graph.graph['node_id']
    for node in graph:
        if b_type == BALANCE:
            if node_ed[node] > 0:
                boundary_nodes.append(node)
        if b_type == REFINE:
            if node_ed[node]-node_id[node] >= 0:
                boundary_nodes.append(node)
    graph.graph['boundary'] = boundary_nodes


def get_refine_gain(graph: Graph, node):
    if len(graph.graph['node_kr_info'][node]) > 0:
        return graph.graph['node_ed'][node]/math.sqrt(len(graph.graph['node_kr_info'][node])) - graph.graph['node_id'][node]
    else:
        return -graph.graph['node_id'][node]


def greedy_k_way_opt(graph: Graph, ctrl: Ctrl, n_iter, f_factor, mode):
    tar_p_wgt = 1/ctrl.nparts*graph.graph['sum_val']
    min_p_wgt = tar_p_wgt*(1/ctrl.un_factor)
    max_p_wgt = tar_p_wgt*ctrl.un_factor
    queue = []
    node_status = {}
    log.debug('Optimize {}, n_iter: {}, f_factor: {}'.format(mode,n_iter, f_factor))
    for i in range(n_iter):
        if mode == BALANCE and max(graph.graph['p_vals']) > max_p_wgt:
            break
        old_cut = graph.graph['cut']
        # np.random.seed(i)
        node_kr_info = graph.graph['node_kr_info']
        node_ed = graph.graph['node_ed']
        node_id = graph.graph['node_id']
        p_vals = graph.graph['p_vals']
        for node in ctrl.rng.choice(graph.graph['boundary'], len(graph.graph['boundary']), replace=False):
            p = graph.nodes[node]['belong']
            queue.append(NodeGain(node, get_refine_gain(graph, node)))
            node_status[node] = PRESENT
        n_moved = 0
        heapq.heapify(queue)
        iii = 0
        log.debug('ready to move {} nodes: {}'.format(len(queue),[node_g.node for node_g in queue],indent=200) )
        while True:
            if len(queue) == 0:
                log.info('迭代: {}, 移动: {}'.format(i, n_moved))
                break
            node_gain = heapq.heappop(queue)
            node = node_gain.node
            log.debug('node gain: {}'.format(node_gain.gain))
            node_status[node] = EXTRACTED
            from_p = graph.nodes[node]['belong']
            load = graph.nodes[node]['load']

            if mode == REFINE:
                if node_id[node] > 0 and p_vals[from_p] - load < min_p_wgt:
                    log.debug(
                        'weight of from_part too small, skip {}'.format(node))
                    continue
            else:
                if p_vals[from_p]-load < min_p_wgt:
                    log.debug(
                        'weight of from_part too small, skip {}'.format(node))
                    continue

            if ctrl.contiguous and is_articulation_node(graph, node):
                log.debug('is_articulation, skip {}'.format(node))
                continue

            if mode == REFINE:
                k = len(node_kr_info[node])
                target_p = None
                for p in node_kr_info[node]:
                    gain = node_kr_info[node][p] - node_id[node]
                    if target_p is None:
                        if gain >= 0 and p_vals[p]+graph.nodes[node]['load'] <= max_p_wgt + f_factor*gain:
                            target_p = p
                    else:
                        if (node_kr_info[node][p] > node_kr_info[node][target_p] and p_vals[p] + graph.nodes[node]['load'] <= max_p_wgt+f_factor*gain) or (
                            node_kr_info[node][p] == node_kr_info[node][target_p] and p_vals[p] < p_vals[target_p]
                        ):
                            target_p = p
                if target_p is None:
                    log.debug('can\'t find target_p, skip {}'.format(node))
                    continue
                gain = node_kr_info[node][target_p] - node_id[node]
                if not (
                    gain > 0 or (gain == 0 and (
                        p_vals[from_p] >= max_p_wgt or
                        p_vals[from_p] > p_vals[tar_p] + graph.nodes[node]['load'] or
                        iii % 2 == 0
                    ))
                ):
                    log.debug('no gain and no better balance, skip {}'.format(node))
                    continue
            else:
                k = len(node_kr_info[node])
                target_p = None
                for p in node_kr_info[node]:
                    if target_p is None:
                        if p_vals[p]+graph.nodes[node]['load'] <= max_p_wgt or p_vals[p]+graph.nodes[node]['load'] <= p_vals[from_p]:
                            target_p = p
                    else:
                        if p_vals[p] < p_vals[target_p]:
                            target_p = p
                if target_p is None:
                    log.debug('can\'t find target_p, skip {}'.format(node))
                    continue
                gain = node_kr_info[node][target_p] - node_id[node]
                if (p_vals[from_p] < max_p_wgt and p_vals[target_p] > min_p_wgt and gain < 0):
                    log.debug('minus gain and no better balance, skip {}'.format(node))
                    continue

            graph.graph['cut'] -= (node_kr_info[node]
                                   [target_p] - node_id[node])
            n_moved += 1
            log.info('remove {} from {} to {}'.format(node, from_p, target_p))
            graph.graph['p_vals'][from_p] -= graph.nodes[node]['load']
            graph.graph['p_vals'][target_p] += graph.nodes[node]['load']
            update_moved_node_info(graph, node, from_p, target_p, mode)
            for nei in graph.neighbors(node):
                nei_p = graph.nodes[nei]['belong']
                num_nei_p_old = len(node_kr_info[nei])
                edge_wei = graph.edges[(node, nei)]['wei']
                update_adjacent_node_info(
                    graph, nei, nei_p, from_p, target_p, edge_wei, mode)
                update_queue_info(graph, queue, node_status, nei,
                                  nei_p, from_p, target_p, num_nei_p_old, mode)
            iii += 1
        for node in node_status:
            node_status[node] = NOT_PRESENT
        if n_moved == 0 or (mode == REFINE and graph.graph['cut'] == old_cut):
            break


def update_moved_node_info(graph: Graph, node, from_p, target_p, mode):
    graph.nodes[node]['belong'] = target_p
    node_kr_info = graph.graph['node_kr_info']
    graph.graph['node_ed'][node] += (graph.graph['node_id']
                                     [node]-node_kr_info[node][target_p])
    graph.graph['node_id'][node], node_kr_info[node][from_p] = node_kr_info[node][target_p], graph.graph['node_id'][node]
    node_kr_info[node].pop(target_p)
    if node_kr_info[node][from_p] == 0:
        node_kr_info[node].pop(from_p)
    boundary = graph.graph['boundary']
    if mode == REFINE:
        if node in boundary and graph.graph['node_ed'][node] - graph.graph['node_id'][node] < 0:
            boundary.remove(node)
        if node not in boundary and graph.graph['node_ed'][node] - graph.graph['node_id'][node] >= 0:
            boundary.append(node)
    else:
        if node in boundary and graph.graph['node_ed'][node] <= 0:
            boundary.remove(node)
        if node not in boundary and graph.graph['node_ed'][node] > 0:
            boundary.append(node)


def update_adjacent_node_info(graph: Graph, nei, nei_p, from_p, target_p, edge_wei, mode):
    node_ed = graph.graph['node_ed']
    node_id = graph.graph['node_id']
    node_kr_info = graph.graph['node_kr_info']
    if nei_p == from_p:
        node_ed[nei] += edge_wei
        node_id[nei] -= edge_wei
        boundary = graph.graph['boundary']
        if mode == REFINE:
            if nei not in boundary and graph.graph['node_ed'][nei] - graph.graph['node_id'][nei] >= 0:
                boundary.append(nei)
        else:
            if nei not in boundary and graph.graph['node_ed'][nei] > 0:
                boundary.append(nei)
    elif nei_p == target_p:
        node_ed[nei] -= edge_wei
        node_id[nei] += edge_wei
        boundary = graph.graph['boundary']
        if mode == REFINE:
            if nei in boundary and graph.graph['node_ed'][nei] - graph.graph['node_id'][nei] < 0:
                boundary.remove(nei)
        else:
            if nei in boundary and graph.graph['node_ed'][nei] < 0:
                boundary.remove(nei)

    if nei_p != from_p:
        if node_kr_info[nei][from_p] == edge_wei:
            node_kr_info[nei].pop(from_p)
        else:
            node_kr_info[nei][from_p] -= edge_wei
    if nei_p != target_p:
        if target_p in node_kr_info[nei]:
            node_kr_info[nei][target_p] += edge_wei
        else:
            node_kr_info[nei][target_p] = edge_wei


def update_queue_info(graph: Graph, queue, node_status, nei, nei_p, from_p, target_p, num_nei_p_old, mode):
    node_ed = graph.graph['node_ed']
    node_id = graph.graph['node_id']
    node_kr_info = graph.graph['node_kr_info']
    if nei_p == target_p or nei_p == from_p or num_nei_p_old != len(node_kr_info[nei]):
        if len(node_kr_info[nei]) > 0:
            r_gain = node_ed[nei] / \
                math.sqrt(len(node_kr_info[nei])) - node_id[nei]
        else:
            r_gain = -node_id[nei]
        if mode == REFINE:
            if node_status[nei] == PRESENT:
                if node_ed[nei] - node_id[nei] >= 0:
                    queue.remove(NodeGain(nei, 1))
                    queue.append(NodeGain(nei, r_gain))
                else:
                    queue.remove(NodeGain(nei, 1))
                    node_status[nei] = NOT_PRESENT
            elif node_status == NOT_PRESENT and node_ed[nei] - node_id[nei] >= 0:
                queue.append(NodeGain(nei, r_gain))
                node_status[nei] = PRESENT
        else:
            if node_status[nei] == PRESENT:
                if node_ed[nei] > 0:
                    queue.remove(NodeGain(nei, 1))
                    queue.append(NodeGain(nei, r_gain))
                else:
                    queue.remove(NodeGain(nei, 1))
                    node_status[nei] = NOT_PRESENT
            elif node_status == NOT_PRESENT and node_ed[nei] > 0:
                queue.append(NodeGain(nei, r_gain))
                node_status[nei] = PRESENT

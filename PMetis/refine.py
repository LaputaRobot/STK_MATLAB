import queue
import numpy as np
import heapq

from config import *
from networkx import Graph
from util import *
from contig import *


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


@get_time
def compute_kway_partition_params(graph: Graph):
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


def project_k_way_partition(ctrl: Ctrl, graph: Graph):
    c_graph = graph.graph['coarser']
    for node in c_graph:
        contains = c_graph.nodes[node]['contains']
        for con_node in contains:
            graph.nodes[con_node]['belong'] = c_graph.nodes[node]['belong']


def RefineKWay(graph: Graph, o_graph: Graph, ctrl: Ctrl):
    # TODO 完成K路划分
    num_comps = find_components(graph)
    ctrl.contiguous = contiguous
    n_level = graph.graph['level']
    if contiguous and num_comps > nparts:
        eliminate_components(graph, ctrl)
        compute_K_way_boundary(graph, ctrl, BALANCE)
        greedy_K_way_optimize(graph, ctrl, 5, 0, BALANCE)
        compute_K_way_boundary(graph, ctrl, REFINE)
        greedy_K_way_optimize(graph, ctrl, ctrl.niter, 0, REFINE)
        ctrl.contiguous = False
    i = 0
    while True:
        if 2*i >= n_level and not is_balanced(ctrl, graph, 0.02):
            compute_K_way_boundary(graph, ctrl, BALANCE)
            greedy_K_way_optimize(graph, ctrl, 1, 0, BALANCE)
            compute_K_way_boundary(graph, ctrl, REFINE)
        greedy_K_way_optimize(graph, ctrl, ctrl.niter, 5, REFINE)
        if contiguous and i == n_level / 2:
            if find_components(graph) > nparts:
                eliminate_components(graph, ctrl)
            if not is_balanced(ctrl, graph, 0.02):
                ctrl.contiguous = 1
                compute_K_way_boundary(graph, ctrl, BALANCE)
                greedy_K_way_optimize(graph, ctrl, 5, 0, BALANCE)
                compute_K_way_boundary(graph, ctrl, REFINE)
                greedy_K_way_optimize(graph, ctrl, ctrl.niter, 0, REFINE)
                ctrl.contiguous = 0
        i += 1
        if graph == o_graph:
            break
        graph = graph.graph['finer']
        project_k_way_partition(ctrl, graph)
        compute_kway_partition_params(graph)
    ctrl.contiguous = contiguous
    if contiguous and find_components(graph) > nparts:
        eliminate_components(graph, ctrl)
    if not is_balanced(ctrl, graph, 0):
        compute_K_way_boundary(graph, ctrl, BALANCE)
        greedy_K_way_optimize(graph, ctrl, 5, 0, BALANCE)
        compute_K_way_boundary(graph, ctrl, REFINE)
        greedy_K_way_optimize(graph, ctrl, ctrl.niter, 0, REFINE)


def is_balanced(ctrl: Ctrl, graph: Graph, f_factor):
    return compute_load_imbalance(graph, nparts) <= f_factor


def compute_load_imbalance(graph: Graph, n):
    p_vals = graph.graph['p_vals']
    max_p_vals = max(p_vals)
    sum_val = sum(p_vals)
    tar_p_val = sum_val/n
    pij = 1/sum_val/(1/n)
    ub_vec = pow(un_factor, 1/math.log(nparts))
    return max_p_vals*pij-ub_vec


def compute_K_way_boundary(graph: Graph, ctrl: Ctrl, b_type):
    boundary_nodes = []
    node_ed = graph.graph['node_ed']
    node_id = graph.graph['node_id']
    for node in graph:
        if b_type == 'balance':
            if node_ed[node] > 0:
                boundary_nodes.append(node)
        if b_type == 'refine':
            if node_ed[node]-node_id[node] >= 0:
                boundary_nodes.append(node)
    graph.graph['boundary'] = boundary_nodes


def get_refine_gain(graph: Graph, node):
    if len(graph.graph['node_kr_info'][node]) > 0:
        return graph.graph['node_ed'][node]/math.sqrt(len(graph.graph['node_kr_info'][node])) - graph.graph['node_id'][node]
    else:
        return -graph.graph['node_id'][node]


def greedy_K_way_optimize(graph: Graph, ctrl: Ctrl, n_iter, f_factor, mode):
    tar_p_wgt = 1/nparts*graph.graph['sum_val']
    min_p_wgt = tar_p_wgt*(1/un_factor)
    max_p_wgt = tar_p_wgt*un_factor
    queue = []
    node_status = {}
    for i in range(n_iter):
        if mode == BALANCE and max(graph.graph['p_vals']) > max_p_wgt:
            break
        old_cut = graph.graph['cut']
        np.random.seed(i)
        node_kr_info = graph.graph['node_kr_info']
        node_ed = graph.graph['node_ed']
        node_id = graph.graph['node_id']
        p_vals = graph.graph['p_vals']
        for node in np.random.choice(graph.graph['boundary'], len(graph.graph['boundary']), replace=False):
            p = graph.nodes[node]['belong']
            queue.append(NodeGain(node, get_refine_gain(graph, node)))
            node_status[node] = PRESENT
        n_moved = 0
        heapq.heapify(queue)
        iii = 0
        while True:
            if len(queue) == 0:
                log.info('迭代: {}, 移动: {}'.format(i,n_moved))
                break
            
            node = heapq.heappop(queue).node
            node_status[node] = EXTRACTED
            from_p = graph.nodes[node]['belong']
            load = graph.nodes[node]['load']

            if mode == REFINE:
                if node_id[node] > 0 and p_vals[from_p] - load < min_p_wgt:
                    log.debug('node_id[node] > 0跳过node {}'.format(node))
                    continue
            else:
                if p_vals[from_p]-load < min_p_wgt:
                    log.debug('other跳过node {}'.format(node))
                    continue

            if contiguous and is_articulation_node(graph, node):
                log.debug('contiguous跳过node {}'.format(node))
                continue

            if mode == REFINE:
                k = len(node_kr_info[node])
                target_p = None
                for p in node_kr_info[node]:
                    gain = node_kr_info[node][p] - node_id[node]
                    if target_p is None:
                        if gain >= 0 and p_vals[p]+graph.nodes[node]['load'] <= max_p_wgt+f_factor*gain:
                            target_p = p
                    else:
                        if (node_kr_info[node][p] > node_kr_info[node][target_p] and p_vals[p]+graph.nodes[node]['load'] <= max_p_wgt+f_factor*gain) or (
                            node_kr_info[node][p] == node_kr_info[node][target_p] and p_vals[p] < p_vals[target_p]
                        ):
                            target_p = p
                if target_p is None:
                    continue
                gain = node_kr_info[node][target_p] - node_id[node]
                if not (
                    gain > 0 or (gain == 0 and (
                        p_vals[from_p] >= max_p_wgt or
                        p_vals[from_p] > p_vals[tar_p] + graph.nodes[node]['load'] or
                        iii % 2 == 0
                    ))
                ):
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
                    continue
                gain = node_kr_info[node][target_p] - node_id[node]
                if (p_vals[from_p] < max_p_wgt and p_vals[target_p] > min_p_wgt and gain < 0):
                    continue

            graph.graph['cut'] -= (node_kr_info[node]
                                   [target_p] - node_id[node])
            n_moved += 1
            log.info('remove {} from {} to {}'.format(node,from_p, target_p))
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

import networkx as nx
import copy

from util import *
from networkx import Graph
from config import *
from config import contiguous_log as log
from pprint import pprint, pformat


def find_components(graph: Graph):
    part = graph.graph['part']
    comp_sum = 0
    for p in part:
        p_nodes = part[p]
        sub_g = graph.subgraph(p_nodes)
        num_comp = nx.number_connected_components(sub_g)
        comp_sum += num_comp
    return comp_sum


def is_articulation_node(graph: Graph, node):
    p = graph.nodes[node]['belong']
    p_nodes = copy.deepcopy(graph.graph['part'][p])
    p_nodes.remove(node)
    if len(p_nodes)==0:
        return True
    sub_g = graph.subgraph(p_nodes)
    return not nx.is_connected(sub_g)


@get_time
def eliminate_components(graph: Graph, ctrl: Ctrl):
    part = graph.graph['part']
    comps = []
    comp_p = []
    comp_w = []
    # 获取component的权重，并将最大权重的component作为原始part
    for p in part:
        p_nodes = part[p]
        sub_g = graph.subgraph(p_nodes)
        comp = list(nx.connected_components(sub_g))
        best_c = None
        max_c_wgt = 0
        for c in comp:
            comps.append(c)
            c_wgt = sum(graph.nodes[n]['load'] for n in c)
            comp_w.append(c_wgt)
            if c_wgt > max_c_wgt:
                max_c_wgt = c_wgt
                best_c = c
        for c in comp:
            if best_c == c:
                comp_p.append(p)
            else:
                comp_p.append(-p)
    log.debug('组件如下: \n{}'.format(pformat_with_indent(comps, width=200)))
    log.debug('comp_w：{}'.format(comp_w))
    # 获取节点所属component
    node_c = {}
    for i in range(len(comps)):
        for node in comps[i]:
            node_c[node] = i
    for c_id in range(len(comps)):
        if comp_p[c_id] < 0:
            # 对需要迁移的component，根据连接度在相邻part选择候选集，并在候选集中找到最平衡的，作为target
            c = comps[c_id]
            c_w = comp_w[c_id]
            nei_comp_connect = {}
            for node in c:
                for nei in graph.neighbors(node):
                    nei_c_id = node_c[nei]
                    # 不是一个区域且无需移动的component
                    if nei_c_id != c_id and comp_p[nei_c_id] >= 0:
                        if nei_c_id in nei_comp_connect:
                            nei_comp_connect[nei_c_id] += graph.edges[node, nei]['wei']
                        else:
                            nei_comp_connect[nei_c_id] = graph.edges[node,
                                                                     nei]['wei']
            max_degree = max(nei_comp_connect.items(), key=lambda x: x[1])
            can_nei_c = [c for c in nei_comp_connect if nei_comp_connect[c]
                         >= max_degree[1]/2]  # 选择候选集
            target = comp_p[max_degree[0]]
            for nei_c_id in can_nei_c:
                cur = comp_p[nei_c_id]
                if target is None or better_balance(c_w, graph.graph['p_vals'][target], graph.graph['p_vals'][cur], graph, ctrl):
                    target = cur
            log.info('move component {} (wight: {})[{}] from part {} to part {}'.format(
                c_id, comp_w[c_id], comps[c_id], -comp_p[c_id], target))
            move_component(graph, ctrl, c_id, target, comps, comp_p, comp_w)
            log.info('cut: {:7.4f}'.format(graph.graph['cut']))
            log.debug('part: \n {}'.format(pformat_with_indent(graph.graph['part'], width=200)))
            log.info('part_val: \n{}'.format(pformat_with_indent(graph.graph['p_vals'], compact=True)))  


def better_balance(c_w, p1_w, p2_w, graph: Graph, ctrl):
    nrm1 = 0
    nrm2 = 0
    max1 = 0
    max2 = 0
    pij = 1/graph.graph['sum_val']/(1/ctrl.nparts)

    tmp1 = pij*(p1_w+c_w) - ctrl.un_factor
    nrm1 += tmp1*tmp1
    max1 = max(max1, tmp1)

    tmp2 = pij*(p2_w + c_w) - ctrl.un_factor
    nrm2 += tmp2*tmp2
    max2 = max(max2, tmp2)
    if max2 < max1:
        return True
    if max2 == max1 and nrm2 < nrm1:
        return True
    return False


def move_component(graph: Graph, ctrl: Ctrl, c_id, target, comps, comp_p, comp_w):
    from k_refine import compute_k_way_params
    for node in comps[c_id]:
        graph.nodes[node]['belong'] = target
    comp_p[c_id] = target
    compute_k_way_params(graph)

import copy
import math
import queue
import time
import copy
import networkx as nx
import matplotlib.pyplot as plt

from networkx import Graph
from k_refine import refine_k_way, compute_k_way_params
from util import Common, get_time, pformat_with_indent,get_log_handlers
from coarsen import *
from pmetis import *
from pprint import pprint
from config import  log
from analysis_result import analysis


def gen_init_graph(graph, link_loads):
    '''
    生成原始图
    :param graph: 传入的图
    :param link_loads: 链路负载
    :return:
    '''
    new_g = Graph(level=0)
    for node in graph:
        new_g.add_node('{}'.format(node[3:]), load=graph.nodes[node]['load'], real_load=0, contains=[], belong='',
                       inner_edge_wei=0)
    for edge in graph.edges:
        new_g.add_edge('{}'.format(edge[0][3:]), '{}'.format(
            edge[1][3:]), wei=link_loads[edge])
    return new_g


def init_KWay_partition(graph: Graph, ctrl: Ctrl):
    log.info('*'*40+' begin init kway partion '+'*'*40)
    ctrl.nCuts = 4
    ctrl.coarsenTo = 20
    ctrl.niter = 10
    graph.graph['sum_val'] = sum(graph.nodes[node]['load'] for node in graph)
    cut = M_level_recur_bisect(graph, graph, ctrl, ctrl.nparts, 0)
    log.info('finish init part, cut: {:7.4f}, bal: {:4.3f}'.format(
        cut, compute_load_imbalance(graph, ctrl.nparts, ctrl)))
    log.info('*'*40+' finish init kway partion '+'*'*40)

def run_py_metis_con(src_common:Common,ctrl:Ctrl, resultLogFile):
    common = src_common
    common.graph = copy.deepcopy(src_common.graph)
    parts=run_metis_main(common, ctrl)
    assignment = {}
    for part in parts:
        assignment['LEO'+parts[part][0]]=['LEO{}'.format(n) for n in parts[part]]
    logger = logging.getLogger('{}'.format(resultLogFile))
    logger.setLevel(logging.INFO)
    handlers = get_log_handlers([LogToFile], resultLogFile)
    for handler in handlers:
        logger.addHandler(handler)
    analysis(common, assignment, logger)

def run_metis_main(common: Common,ctrl: Ctrl):
    origin_graph = gen_init_graph(common.graph, common.link_load)
    ctrl.coarsenTo = max(origin_graph.number_of_nodes() /
                         20 * math.log(ctrl.nparts), 30 * ctrl.nparts)
    ctrl.nIparts = 4 if ctrl.coarsenTo == 30 * ctrl.nparts else 5
    log.info('args: \n{}'.format(pformat_with_indent(dict(ctrl))))
    coarsest_graph = coarsen_graph(origin_graph, ctrl)

    init_KWay_partition(coarsest_graph, ctrl)
    part, part_val = compute_k_way_params(coarsest_graph)
    log.debug('part: \n {}'.format(pformat_with_indent(part, width=200)))
    log.info('part_val: \n{}'.format(pformat_with_indent(part_val, compact=True)))    

    refine_k_way(coarsest_graph, origin_graph, ctrl)
    log.info('part: \n {}'.format(pformat_with_indent(origin_graph.graph['part'], width=100, compact=True)))
    log.info('part_val: \n{}'.format(pformat_with_indent(origin_graph.graph['p_vals'], compact=True))) 
    
    return origin_graph.graph['part']
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

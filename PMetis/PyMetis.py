import copy
import math
import queue
import time

import matplotlib.pyplot as plt
from networkx import Graph
import networkx as nx
from config import nparts, MatchOrder, MatchScheme
from util import Common
from coarsen import *
from pmetis import *


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
        new_g.add_edge('{}'.format(edge[0][3:]), '{}'.format(edge[1][3:]), wei=link_loads[edge])
    return new_g


def init_KWay_partitioning(graph: Graph, ctrl: Ctrl):
    ctrl.nCuts = 4
    ctrl.coarsenTo = 20
    M_level_recursive_bisection(graph, ctrl, nparts, 0)


def run_metis_main(common: Common):
    origin_graph = gen_init_graph(common.graph, common.link_load)
    ctrl = Ctrl()
    ctrl.coarsenTo = max(origin_graph.number_of_nodes() / 20 * math.log(nparts), 30 * nparts)
    ctrl.nIparts = 4 if ctrl.coarsenTo == 30 * nparts else 5
    coarsest_graph = coarsen_graph(origin_graph, ctrl)
    init_KWay_partitioning(coarsest_graph, ctrl)
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

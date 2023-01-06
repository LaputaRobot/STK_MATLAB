import copy
import heapq
import logging
import math
import os.path
import queue
import random
import sys
import time

import networkx as nx
import numpy.random
from matplotlib import pyplot as plt
from networkx import Graph
from numpy.random import default_rng
from  PyMetis import edge_equal
from getSatLoad import getLoad


# from util import draw_result_with_time, get_lbr


def testNetworkx():
    graph = Graph()
    graph.add_node('11', load=12)
    graph.nodes['11']['load'] += 1
    graph.add_edge('1', '2')
    graph.add_edge('1', '3')

    # print(graph.nodes['11']['load'])
    x = (2, 1)
    print(x[1])
    # print(set(list(graph.edges)[0],list(graph.edges)[0]))


def deprecatedFun():
    print("this is deprecated!!!")


def gen_result():
    result = {}
    for i in range(1, 9):
        result['LEO{}{}'.format(i, i)] = []
        for j in range(1, 10):
            result['LEO{}{}'.format(i, i)].append('LEO{}{}'.format(i, j))
    print(result)


def print_x():
    x = 1
    print(x)


def test_metis_bi_result(xadj_file, adjncy_file, where_file):
    with open(xadj_file, 'r') as f:
        xadj_lines = f.readlines()
    with open(adjncy_file, 'r') as f:
        adjncy_lines = f.readlines()
    with open(where_file, 'r') as f:
        where_lines = f.readlines()
    graph = Graph()
    xadj = []
    adjncy = []
    where = []
    for l in xadj_lines:
        xadj.append(int(l.split(' ')[-1]))
    for l in adjncy_lines:
        adjncy.append(int(l.split(' ')[-1]))
    for l in where_lines:
        where.append(int(l.split(' ')[-1]))
    partitions = set()
    for node in range(len(xadj)):
        if node == len(xadj) - 1:
            continue
        partitions.add(where[node])
        graph.add_node(node, p=where[node])
        # print(node)
        for nei in adjncy[xadj[node]:xadj[node + 1]]:
            # print(nei,end=', ')
            graph.add_node(nei)
            graph.add_edge(node, nei)
    nodes = list(graph.nodes)
    num_comps = 0
    for p in partitions:
        new_graph = copy.deepcopy(graph)
        for node in nodes:
            if new_graph.nodes[node]['p'] != p:
                new_graph.remove_node(node)
        comp = nx.components.number_connected_components(new_graph)
        num_comps += comp
        print(comp)
    print(num_comps)
    # pos = nx.drawing.spring_layout(graph,seed=2)
    # labels = {n: '{}'.format(n) for n in pos}
    # colors=[]
    # rng=default_rng(1)
    # for p in partitions:
    #     color=[]
    #     for i in range(3):
    #         color.append(rng.uniform(0,1))
    #     colors.append(color)
    # node_colors = []
    # for n in pos:
    #     node_colors.append(colors[graph.nodes[n]['p']])
    # nx.draw(graph, pos, labels=labels, with_labels=True, node_color=node_colors)
    # plt.show()

    # plt.figure(figsize=(11, 8), dpi=300)
    # nx.draw(graph,with_labels=True)
    # plt.show()
    # print(graph.number_of_nodes())
    # print(graph.number_of_edges())


def testPygMetis():
    pass


class C:
    def __init__(self, x):
        self.x = x

    def printx(self):
        print(self.x)


if __name__ == '__main__':
    g = Graph(name='graph')
    print(g.graph['name'])
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    for e in g.edges():
        print(edge_equal(e, (1, 2)))
    # xadj_file = ' xadj.txtxxx'
    # adjncy_file = 'adjncy.txtxxx'
    # where_file = 'where.txt'
    # test_metis_bi_result(xadj_file, adjncy_file, where_file=where_file)
    # draw_result_with_time('my', 1330, 'load')
    # draw_result_with_time('src', 1330, 'load')
    # loads1={'LEO49': 51.25823719895647, 'LEO32': 54.506696195103686, 'LEO37': 18.26353650454407, 'LEO33': 31.287744452281707, 'LEO58': 42.0293023981377, 'LEO61': 58.30877457803851, 'LEO73': 14.29823123002132, 'LEO77': 10.306880629680094}
    # loads2={'LEO39': 37.68074349354527, 'LEO12': 22.31773215773854, 'LEO35': 24.832627632970464, 'LEO32': 54.96015432072758, 'LEO49': 45.26250853502802, 'LEO61': 49.00406262092618, 'LEO58': 22.019914326348154, 'LEO69': 32.61565351541098}
    # print(get_lbr(list(loads1.values())))
    # print(get_lbr(list(loads2.values())))
    # for i in range(20,110,5):
    #     if i!=100:
    #         print('{}, {}'.format(i,1 / (100 - i) * 1000))
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(11, 8), dpi=100)
    # plt.plot([1,2,3,4,5])
    # plt.title("x")
    # plt.show()
    # print(int(22/10)+1)
    # args='file.uf.con'
    # logger = logging.getLogger('mylogger')
    # logger.setLevel(logging.INFO)
    # f_handler = logging.FileHandler('metisResult/my/{}'.format(args))
    # logger.addHandler(f_handler)
    # logger.info('ok')
    # logger.info('ok')
    # print('result: {:->10.2f}'.format(1/3))
    # print('result: {:4d}'.format(13))
    # d = {1: 2, 2: 2}
    # d1 = dict([(1, 2)])
    # print(d1.has_key(1))
    # s = sorted(link_loads.items(), key=lambda x: x[1], reverse=True)
    # s_dict = dict(s)
    # src=s[0][0][0]
    # dst=s[0][0][1]
    # print(src,dst)
    # print(s_dict)
    # sorted_neighbor_links_list = sorted({}.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_neighbor_links_list)
    # print(sorted([2,1]))

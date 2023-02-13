import copy
import heapq
import logging
import math
import os.path
import queue
import random
import sys
import time
import numpy as np
from pprint import pprint,pformat
import networkx as nx
import numpy.random
import multiprocessing

from matplotlib import pyplot as plt
from networkx import Graph
from numpy.random import default_rng
from PyMetis import edge_equal
from getSatLoad import getLoad
from config import *
from pmetis import *
from util import pformat_with_indent,get_time
from concurrent.futures import ThreadPoolExecutor,as_completed

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


def fun1(graph):
    return Graph()

def testlog():
    log=get_logger()
    log.info("msg")

def assert_diff_func(n):
    p_vals=[16,15,13,14,18,21,13,18]
    max_p_vals = max(p_vals)
    sum_val = sum(p_vals)
    tar_p_val = sum_val/n
    pij = (1/sum_val)/(1/n)
    ub_vec = 1.3
    # ub_vec = pow(un_factor, 1/math.log(nparts))
    print(max_p_vals*pij-ub_vec)

class kv(object):
    def __init__(self, k,v):
        self.k=k
        self.v=v
    
    def __lt__(self, other):
        return self.v > other.v

    def __str__(self):
        return "{{{}: {}}}".format(self.k,self.v) 

    def __eq__(self, other):
        return self.k == other.k

def testPQueue():
    q=[]
    for i in range(20):
        item=kv('{}'.format(i),random.randint(1, 100))
        q.append(item)
        print('add: ', item)
    for i in range(10):
        q.remove(kv('{}'.format(i),1))
    heapq.heapify(q)
    while len(q)>0:
        item=heapq.heappop(q)
        print(item)

def fun1():
    p=1

def test_grammar():
    dic={'name':'y','age':12}
    dic_copy=dic
    dic_copy['name']='z'

    lis=[1,2,3]
    lis_copy=lis
    # lis_copy[0]=2
    # print(lis)
    lis[0]=2

    for p in range(4,5):
        for i in range(10):
            print(p)
            fun1()
    


def test_log():
    # from config import coarsen_log as log
    logger = logging.getLogger('test')
    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter('%(filename)-15s:%(lineno)d - %(funcName)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('info')
    logger.debug('debug')
    print('print')
    # import util.log as n_log
    # log.debug('debug')
    # log.info('info')

def get_html(times):
    time.sleep(times%2)
    # print("get page {}s finished".format(times))
    return times

def test_pprint():
    s = pformat([i for i in range(20)], indent=20,width=30,compact=True)
    print(s)


def test_thread_pool():
    executor=ThreadPoolExecutor(max_workers=50)
    tasks=[executor.submit(get_html,i) for i in range(50)]
    for f in as_completed(tasks):
        print('{}: {} done'.format(time.asctime() ,f.result()))

@get_time
def test_get_time():
    # t1=time.time()
    sum=0
    for i in range(992):
        for j in range(5):
            sum+=1
    # print(time.time()-t1)

def put_num(i,ass_l):
    x = random.randint(1, 10)
    time.sleep(0.1)
    if x not in ass_l:
        # lock.acquire()
        print('{} add {}'.format(i,x))
        # ass_l.append(x)
        ass_l[x] =1
        # lock.release()
    else:
        print('skip: {}'.format(i))
    

def test_multiprocess():
    pool=multiprocessing.Pool(processes=5)
    lock=multiprocessing.Lock()
    ass_set= set()
    for i in range(1000):
        pool.apply_async(put_num,(i,ass_set, lock,))
    pool.close()
    pool.join()  
    
def printi(i):
    print(i)

if __name__ == '__main__':
    # test_multiprocess()
    lock=multiprocessing.Lock()
    d=multiprocessing.Manager().dict()
    pool=multiprocessing.Pool(processes=4)
    for i in range(100):
        pool.apply_async(put_num,(i, d,))
        # pool.apply_async(printi,(i,))
    pool.close()
    pool.join() 
    print(d)

    # test_get_time()
    # testlog()
    # assert_diff_func(8)
    # testPQueue()
    # test_grammar()
    # test_pprint()
    # test_log()
    # test_thread_pool()
    # pprint_with_indent([i for i in range(20)], width=30,compact=True)
    # print(os.cpu_count())
    # G = nx.Graph()
    # G.add_edge(0, 1)
    # G.add_edge(0, 2, wei=3)
    # print(G.number_of_nodes())
    # print(G.has_edge(2, 0))
    # G.add_edge(2,0, wei=4)
    # print(G.number_of_nodes())
    # neis = nx.neighbors(G, 0)
    # print(len(list(neis)))
    # print(G.neighbors(0))
    # print('degree: {}'.format(G.degree(0, weight='wei')))

    # lis = [1, 3, 6, 5]
    # lis1 = lis
    # lis1[0] = 3
    # x = list(filter(lambda x: x > 3, lis))
    # print(x)
    # print(lis)
   
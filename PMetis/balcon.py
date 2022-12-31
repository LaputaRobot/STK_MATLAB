import copy
import math
import time

from PMetis.config import MCS, MSSLS
from PMetis.analysis_result import apply_partition
from PMetis.util import Common
import networkx as nx


def bal_con_assign(common: Common, initial_assign):
    start_time = time.time()
    iteration = 1
    assign = initial_assign
    g = common.graph
    pair_load = common.pair_load
    pair_path_delay = common.pair_path_delay
    con_loads = apply_partition(g, common.link_load, assign)
    max_con = max(con_loads, key=con_loads.get)
    max_con_load = con_loads[max_con]
    sum_con_load = sum(con_loads.values())
    last_sum_con_load = math.inf
    last_max_con_load = math.inf
    last_assign = copy.deepcopy(assign)
    while last_sum_con_load > sum_con_load or last_max_con_load > max_con_load:
        last_sum_con_load = sum_con_load
        last_max_con_load = max_con_load
        alternatives = []
        start_switches = assign[max_con]
        for sw in start_switches:
            cluster = [sw]
            alternatives.extend(computeMigrationAl(cluster, assign, max_con, max_con_load, common))
            while True:
                newCluster = increaseCluster(cluster, common)
                if len(newCluster) > MCS or newCluster == cluster:
                    break
                cluster = newCluster
                alternatives.extend(computeMigrationAl(cluster, assign, max_con, max_con_load, common))
            if len(alternatives) > MSSLS:
                break
        best_alter = getBestAl(alternatives)
        new_assign = copy.deepcopy(assign)
        for node in best_alter['cluster']:
            controller = getController(new_assign, node)
            if best_alter['dstCon'] != controller:
                new_assign[controller].remove(node)
                new_assign[best_alter['dstCon']].append(node)
        tmp = {}
        for val in new_assign.values():
            tmp[val[0]] = val
        last_assign = copy.deepcopy(assign)
        assign = tmp
        con_loads = apply_partition(g, common.link_load, assign)
        max_con = max(con_loads, key=con_loads.get)
        max_con_load = con_loads[max_con]
        sum_con_load = sum(con_loads.values())
        #     print('assignment: {}'.format(assign))
        print('{:^5.2f}, iter{:->3d}'.format(time.time() - start_time, iteration), end=', ')
        iteration += 1
        print('SumLoad: {:>7.4f}'.format(sum_con_load), end=', ')
        print('MaxCon: {}, Load: {:>7.4f}'.format(max_con, max_con_load))
    # print(lastAssign)
    # sys.exit(0)
    return last_assign


def computeMigrationAl(cluster, assign, srcCon, srcConLoad, common: Common):
    alternatives = []
    g = common.graph
    pair_load = common.pair_load
    pair_path_delay = common.pair_path_delay
    for con in assign:
        newAssign = copy.deepcopy(assign)
        for sw in cluster:
            controller = getController(newAssign, sw)
            if con != controller:
                newAssign[con].append(sw)
                newAssign[controller].remove(sw)
        newConLoads = apply_partition(g, common.link_load, newAssign)
        if newConLoads[con] < srcConLoad:
            alternative = {'srcCon': srcCon, 'cluster': cluster, 'dstCon': con, 'dstConLoad': newConLoads[con],
                           'sumLoad': sum(newConLoads.values())}
            alternatives.append(alternative)
    return alternatives


def getController(assign, sw):
    for con in assign:
        if sw in assign[con]:
            return con


def getBestAl(alternatives):
    min_sum_load = math.inf
    best_alter = None
    for alter in alternatives:
        if alter['sumLoad'] < min_sum_load:
            best_alter = alter
            min_sum_load = alter['sumLoad']
    return best_alter


def increaseCluster(cluster, common: Common):
    neighborsC = getNeighbor(cluster, common.graph)
    densities = {}
    for neighbor in neighborsC:
        densities[neighbor] = getDensity(cluster, common.link_load, common.graph, extend=[neighbor])
    bestNei = max(densities, key=densities.get)
    newCluster = copy.deepcopy(cluster)
    newCluster.append(bestNei)
    return newCluster


def getNeighbor(cluster, g):
    neighborsC = []
    bestNei = None
    for sw in cluster:
        nei = nx.neighbors(g, sw)
        neighborsC.extend([n for n in nei if n not in cluster])
    return neighborsC


def getDensity(cluster, link_load, g, extend=None):
    if extend is None:
        extend = []
    innerWeight = 0
    outerWeight = 0
    clu = copy.deepcopy(cluster)
    if len(extend) > 0:
        clu.extend(extend)
    neighborsC = getNeighbor(clu, g)
    for x in clu:
        for y in clu:
            if g.has_edge(x, y):
                innerWeight += link_load[(x, y)]
    for neighbor in neighborsC:
        for inner in clu:
            if g.has_edge(neighbor, inner):
                outerWeight += link_load[(neighbor, inner)]
    if outerWeight == 0:
        return 1
    res = innerWeight / (innerWeight + outerWeight)
    return res

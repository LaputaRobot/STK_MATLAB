import copy
import math
import time

from config import MCS, MSSLS
from analysis_result import apply_partition
from util import Common
import networkx as nx


def bal_con_assign(common: Common, initial_assign, logger):
    start_time = time.time()
    iteration = 1
    assign = initial_assign
    g = common.graph
    con_loads = apply_partition(g, common.link_load, assign)
    max_con = max(con_loads, key=con_loads.get)
    max_con_load = con_loads[max_con]
    sum_con_load = sum(con_loads.values())
    sum_con_load_l = math.inf
    max_con_load_l = math.inf
    assign_l = copy.deepcopy(assign)
    while sum_con_load_l > sum_con_load or max_con_load_l > max_con_load:
        sum_con_load_l = sum_con_load
        max_con_load_l = max_con_load
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
        if best_alter is not None:
            for node in best_alter['cluster']:
                controller = getController(new_assign, node)
                if best_alter['dstCon'] != controller:
                    new_assign[controller].remove(node)
                    new_assign[best_alter['dstCon']].append(node)
        tmp = {}
        for val in new_assign.values():
            tmp[val[0]] = val
        assign_l = copy.deepcopy(assign)
        assign = tmp
        con_loads = apply_partition(g, common.link_load, assign)
        max_con = max(con_loads, key=con_loads.get)
        max_con_load = con_loads[max_con]
        sum_con_load = sum(con_loads.values())
        logger.info('{:^5.2f}, iter{:->3d}, '.format(time.time() - start_time, iteration))
        iteration += 1
        logger.info('SumLoad: {:>7.4f}, MaxCon: {}, Load: {:>7.4f} \n'.format(sum_con_load,max_con, max_con_load))
    return assign_l


def computeMigrationAl(cluster, assign, srcCon, srcConLoad, common: Common):
    alternatives = []
    g = common.graph
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

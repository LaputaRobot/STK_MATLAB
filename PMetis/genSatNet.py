import logging
import math
import sys
import time
from itertools import islice
from pprint import pprint
import copy
import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx import Graph, path_weight
from config import *
from util import *
from metis import *


def apply_placement_assignment(graph, pair_load, pair_path_delay, result):
    con_loads = {}
    for con in result:
        con_load = 0
        for sw in result[con]:
            graph.nodes[sw]['controller'] = con
            con_load += graph.nodes[sw]['load']
        for sw in result[con]:
            # print('get sw {} inter-domain load'.format(sw))
            for pair in pair_load:
                if pair_load[pair] != 0:
                    path = pair_path_delay[pair][0]
                    if sw in path:
                        sw_index = path.index(sw)
                        if sw_index > 0 and graph.nodes[sw]['controller'] != graph.nodes[path[sw_index - 1]][
                            'controller']:
                            con_load += pair_load[pair]
        con_loads[con] = con_load
        for sw in result[con]:
            graph.nodes[sw]['con_load'] = con_load
    return con_loads


def get_avg_flow_setup_time(graph, pair_load, pair_path_delay):
    '''
    获取所有流量对的安装时延的平均值

    :param graph: 图
    :param pair_load: 流量对
    :param pair_path_delay: 节点间的时延
    :return: 流安装时延的平均值
    '''
    all_pairs_setup_time = 0
    len_sum=0
    len_num=0
    for pair in pair_load:
        if pair_load[pair] != 0:
            # print('get pair {}'.format(pair))
            setup_time = pair_path_delay[pair][1]
            path = pair_path_delay[pair][0]
            len_sum+=len(path)
            len_num+=1
            # print('get pair path {}'.format(path))
            same_con_switches = []
            for node_index in range(len(path)):
                sw = path[node_index]
                if node_index == 0 or graph.nodes[sw]['controller'] != graph.nodes[path[node_index - 1]]['controller']:
                    propagation_delay = pair_path_delay[(sw, graph.nodes[sw]['controller'])][1]
                    con_process_delay = max(0, 1 / (100 - graph.nodes[path[node_index]]['con_load']) * 1000)
                    setup_time += (propagation_delay + con_process_delay)
                    same_con_switches = [sw]
                    # print('packet in at {}, delay={}+{}'.format(sw, propagation_delay, con_process_delay))
                if node_index == len(path) - 1 or graph.nodes[sw]['controller'] != graph.nodes[path[node_index + 1]][
                    'controller']:
                    install_rule_time = max(
                        [pair_path_delay[(sw, graph.nodes[sw]['controller'])][1] for sw in same_con_switches])
                    setup_time += install_rule_time
                    same_con_switches = []
                    # print('install rule at {}, delay={}'.format(sw, install_rule_time))
                else:
                    same_con_switches.append(sw)
                    # print('go through {}'.format(sw))
            # print('pair {} load {} path {} time {}'.format(pair, pair_load[pair], path, setup_time))
            all_pairs_setup_time += setup_time * pair_load[pair]
    # print(len_sum/len_num)
    return all_pairs_setup_time / sum(pair_load.values())


def get_link_load(graph: Graph, pair_load, pair_path_delay):
    edge_loads = {}
    edges = list(graph.edges)
    load_sum = 0
    for edge in edges:
        src = edge[0]
        dst = edge[1]
        edge_loads[(src, dst)] = 0
        edge_loads[(dst, src)] = 0
    for pair in pair_load:
        if pair_load[pair] != 0:
            path = pair_path_delay[pair][0]
            for index in range(len(path) - 1):
                src = path[index]
                dst = path[index + 1]
                # edge_loads[(src, dst)].append([pair, pair_load[pair]])
                edge_loads[(src, dst)] += pair_load[pair]
                load_sum += pair_load[pair]
    # print(load_sum)
    return edge_loads


def greedy_alg1(graph: Graph):
    # con_cap = 80
    areas = []
    chosen_node = []
    sorted_link_load_dict = dict(sorted(link_load.items(), key=lambda x: x[1], reverse=True))
    # print(sorted_link_load_dict)
    # sys.exit(0)
    area = []
    while sum(len(a) for a in areas) < 72:
        # print(len(chosen_node))
        # print(sum(len(a) for a in areas))
        if len(area) == 0:
            sorted_link_load = sorted(sorted_link_load_dict.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_link_load) == 0:
                break
            # print(sorted_link_load)
            src = sorted_link_load[0][0][0]
            dst = sorted_link_load[0][0][1]
            area.append(src)
            area.append(dst)
            chosen_node.append(src)
            chosen_node.append(dst)
            sorted_link_load_dict.pop((src, dst))
            sorted_link_load_dict.pop((dst, src))
            # print('add {} {} to empty area'.format(src,dst))
            # print('pop {} and {}'.format((src, dst),(dst, src)))
        area_load = 0
        # print('areas: {}'.format(areas))
        for node in area:
            area_load += graph.nodes[node]['load']
            # neighbors = nx.neighbors(graph, node) for nei in neighbors: if nei in area: continue else: area_load +=
            # link_load[(node, nei)] area_load += link_load[(nei, node)] print('add {} load {}, {} load {}'.format(
            # (node, nei),link_load[(node, nei)],(nei, node),link_load[(nei, node)]))
        # print(sorted_link_load_dict)
        if area_load > 15:
            areas.append(area)
            remove_keys = []
            # print('append area {}'.format(area))
            for node in area:
                # print(node)
                for key in sorted_link_load_dict:
                    if node in key and key not in remove_keys:
                        # print('remove key {}'.format(key))
                        remove_keys.append(key)
            for key in remove_keys:
                sorted_link_load_dict.pop(key)
            area = []
            # print('reset area when area load = {}'.format(area_load))
        else:
            neighbor_links = {}
            for node in area:
                neighbors = nx.neighbors(graph, node)
                for nei in neighbors:
                    if nei in area or nei in chosen_node:
                        continue
                    else:
                        neighbor_links[(node, nei)] = link_load[(node, nei)]
                        neighbor_links[(nei, node)] = link_load[(nei, node)]
            sorted_neighbor_links_list = sorted(neighbor_links.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_neighbor_links_list) == 0:
                areas.append(area)
                # print('append area {}'.format(area))
                remove_keys = []
                for node in area:
                    # print(node)
                    for key in sorted_link_load_dict:
                        if node in key and key not in remove_keys:
                            # print('remove key {}'.format(key))
                            remove_keys.append(key)
                for key in remove_keys:
                    sorted_link_load_dict.pop(key)
                # print('reset area {} when there is no neighbor'.format(area))
                area = []
                continue
            src = sorted_neighbor_links_list[0][0][0]
            dst = sorted_neighbor_links_list[0][0][1]
            if src not in area:
                area.append(src)
                chosen_node.append(src)
                # print('add {} to area'.format(src))
            if dst not in area:
                area.append(dst)
                chosen_node.append(dst)
                # print('add {} to area'.format(dst))
            sorted_link_load_dict.pop((src, dst))
            sorted_link_load_dict.pop((dst, src))
            # print('pop {} and {}'.format((src, dst), (dst, src)))
    result = {}
    for area in areas:
        result[area[0]] = area
    unassigned_nodes = []
    copy_graph = copy.deepcopy(graph)
    apply_placement_assignment(copy_graph, pair_load, pair_path_delay, result)
    for node in copy_graph.nodes:
        if copy_graph.nodes[node]['controller'] == '':
            unassigned_nodes.append(node)
    if len(unassigned_nodes) > 0:
        for node in unassigned_nodes:
            neighbors = nx.neighbors(copy_graph, node)
            min_neighbor_con = None
            min_neighbor_con_load = 1000
            for nei in neighbors:
                if copy_graph.nodes[nei]['con_load'] < min_neighbor_con_load:
                    min_neighbor_con_load = copy_graph.nodes[nei]['con_load']
                    min_neighbor_con = copy_graph.nodes[nei]['controller']
            print('assign {} to {}'.format(node, min_neighbor_con))
            result[min_neighbor_con].append(node)
    return result


def deployInArea(G, pair_load, pair_path_delay, link_load, assign):
    apply_placement_assignment(G, pair_load, pair_path_delay, assign)
    node_loads = {}
    copy_graph = copy.deepcopy(G)
    for node in copy_graph.nodes:
        node_loads[node] = copy_graph.nodes[node]['load']
        neighbors = nx.neighbors(copy_graph, node)
        for nei in neighbors:
            if copy_graph.nodes[nei]['controller'] != copy_graph.nodes[node]['controller']:
                node_loads[node] += link_load[(nei, node)]
    new_result = {}
    sum_min_delay=0
    for con in assign:
        area = assign[con]
        min_delay_con = None
        min_delay = 10000
        for newCon in area:
            all_delay = 0
            for sw in area:
                if newCon != sw:
                    all_delay += node_loads[sw] * pair_path_delay[(sw, newCon)][1]
            if all_delay < min_delay:
                min_delay = all_delay
                min_delay_con = newCon
        # print('{} all delay: {}'.format(min_delay_con,min_delay))
        sum_min_delay+=min_delay
        new_result[min_delay_con] = area
    # print('*'*20,'node loads','*'*20, '\n',exch(node_loads))
    # print('sum_min_delay {}'.format(sum_min_delay))
    # logger.info('*' * 20, 'Final Assignment', '*' * 20)
    # return new_result
    return new_result,sum_min_delay,node_loads


def bal_con_assign(initialAssign):
    startTime = time.time()
    iteration = 1
    assign = initialAssign
    conLoads = apply_placement_assignment(G, pair_load, pair_path_delay, assign)
    maxCon = max(conLoads, key=conLoads.get)
    maxConLoad = conLoads[maxCon]
    sumConLoad = sum(conLoads.values())
    lastSumConLoad = math.inf
    lastMaxConLoad = math.inf
    lastAssign = copy.deepcopy(assign)
    while lastSumConLoad > sumConLoad or lastMaxConLoad > maxConLoad:
        lastSumConLoad = sumConLoad
        lastMaxConLoad = maxConLoad
        alternatives = []
        startSwitches = assign[maxCon]
        for sw in startSwitches:
            cluster = [sw]
            alternatives.extend(computeMigrationAl(cluster, assign, maxCon, maxConLoad))
            while True:
                newCluster = increaseCluster(cluster)
                if len(newCluster) > MCS or newCluster == cluster:
                    break
                cluster = newCluster
                alternatives.extend(computeMigrationAl(cluster, assign, maxCon, maxConLoad))
            if len(alternatives) > MSSLS:
                break
        bestAlter = getBestAl(alternatives)
        newAssign = copy.deepcopy(assign)
        for node in bestAlter['cluster']:
            controller = getController(newAssign, node)
            if bestAlter['dstCon'] != controller:
                newAssign[controller].remove(node)
                newAssign[bestAlter['dstCon']].append(node)
        tmp = {}
        for val in newAssign.values():
            tmp[val[0]] = val
        lastAssign = copy.deepcopy(assign)
        assign = tmp
        conLoads = apply_placement_assignment(G, pair_load, pair_path_delay, assign)
        maxCon = max(conLoads, key=conLoads.get)
        maxConLoad = conLoads[maxCon]
        sumConLoad = sum(conLoads.values())
        #     print('assignment: {}'.format(assign))
        print('{:^5.2f}, iter{:->3d}'.format(time.time() - startTime, iteration), end=', ')
        iteration += 1
        print('SumLoad: {:>7.4f}'.format(sumConLoad), end=', ')
        print('MaxCon: {}, Load: {:>7.4f}'.format(maxCon, maxConLoad))
    # print(lastAssign)
    # sys.exit(0)
    return lastAssign


def computeMigrationAl(cluster, assign, srcCon, srcConLoad):
    alternatives = []
    for con in assign:
        newAssign = copy.deepcopy(assign)
        for sw in cluster:
            controller = getController(newAssign, sw)
            if con != controller:
                newAssign[con].append(sw)
                newAssign[controller].remove(sw)
        newConLoads = apply_placement_assignment(G, pair_load, pair_path_delay, newAssign)
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
    minSumLoad = math.inf
    bestAlter = None
    for alter in alternatives:
        if alter['sumLoad'] < minSumLoad:
            bestAlter = alter
            minSumLoad = alter['sumLoad']
    return bestAlter


def increaseCluster(cluster):
    neighborsC = getNeighbor(cluster)
    densities = {}
    for neighbor in neighborsC:
        densities[neighbor] = getDensity(cluster, extend=[neighbor])
    bestNei = max(densities, key=densities.get)
    newCluster = copy.deepcopy(cluster)
    newCluster.append(bestNei)
    return newCluster


def getNeighbor(cluster):
    neighborsC = []
    bestNei = None
    for sw in cluster:
        nei = nx.neighbors(G, sw)
        neighborsC.extend([n for n in nei if n not in cluster])
    return neighborsC


def getDensity(cluster, extend=[]):
    innerWeight = 0
    outerWeight = 0
    clu = copy.deepcopy(cluster)
    if len(extend) > 0:
        clu.extend(extend)
    neighborsC = getNeighbor(clu)
    for x in clu:
        for y in clu:
            if G.has_edge(x, y):
                innerWeight += link_load[(x, y)]
    for neighbor in neighborsC:
        for inner in clu:
            if G.has_edge(neighbor, inner):
                outerWeight += link_load[(neighbor, inner)]
    if outerWeight == 0:
        return 1
    res = innerWeight / (innerWeight + outerWeight)
    return res


def genMetisGraph(time):
    f = open('MetisTopos/{}'.format(time), 'w')
    f.write('{} {} 011\n'.format(len(G.nodes), len(G.edges)))
    for node in G.nodes:
        f.write('{} '.format(G.nodes[node]['load'] * 100))
        neighbors = G.neighbors(node)
        for nei in neighbors:
            f.write('{} '.format(getIndex(nei)))
            if link_load[(nei, node)] == 0:
                f.write('1 ')
            else:
                if getIndex(node) > getIndex(nei):
                    f.write('{:.0f} '.format(link_load[(nei, node)] * 100))
                else:
                    f.write('{:.0f} '.format(link_load[(node, nei)] * 100))
        f.write('\n')
    f.close()


def genMetisGraphwithLink():
    f = open('metisLinkG72', 'w')
    fNamed = open('metisLinkG72-named', 'w')
    fOrder = open('nodeOrder.txt', 'w')
    graph = copy.deepcopy(G)
    additionNodes = {}
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            if graph.has_edge(node1, node2):
                additionNodes['{}-{}'.format(node1, node2)] = link_load[(node1, node2)]
                additionNodes['{}-{}'.format(node2, node1)] = link_load[(node1, node2)]
    for node in additionNodes:
        graph.add_node(node, load=additionNodes[node])
    for node in additionNodes:
        node1 = node.split('-')[0]
        node2 = node.split('-')[1]
        otherNode = '{}-{}'.format(node2, node1)
        graph.add_edge(node, otherNode)
        graph.add_edge(node, node1)
        if graph.has_edge(node1, node2):
            graph.remove_edge(node1, node2)
    f.write('{} {} 011\n'.format(len(graph.nodes), len(graph.edges)))
    fNamed.write('{} {} 011\n'.format(len(graph.nodes), len(graph.edges)))
    nodes = list(graph.nodes)
    fOrder.write(nodes.__str__())
    fOrder.close()
    for node in nodes:
        fNamed.write('{} '.format(node))
        if node in additionNodes:
            f.write('{:.0f} '.format(additionNodes[node] * 100))
            fNamed.write('{:.0f} '.format(additionNodes[node] * 100))
        else:
            f.write('{} '.format(graph.nodes[node]['load'] * 100))
            fNamed.write('{} '.format(graph.nodes[node]['load'] * 100))
        neighbors = graph.neighbors(node)
        for nei in neighbors:
            if len(node) != len(nei):
                f.write('{} {} '.format(nodes.index(nei) + 1, 100000))
                fNamed.write('{} {} '.format(nei, 100000))
            else:
                if len(node) < 6:
                    continue
                else:
                    node1 = node.split('-')[0]
                    node2 = node.split('-')[1]
                    if link_load[(node1, node2)] > 0:
                        if nodes.index(node1) < nodes.index(node2):
                            f.write('{} {:.0f} '.format(nodes.index(nei) + 1, link_load[(node1, node2)] * 100))
                            fNamed.write('{} {:.0f} '.format(nei, link_load[(node1, node2)] * 100))
                        else:
                            f.write('{} {:.0f} '.format(nodes.index(nei) + 1, link_load[(node2, node1)] * 100))
                            fNamed.write('{} {:.0f} '.format(nei, link_load[(node2, node1)] * 100))
                    else:
                        f.write('{} 1 '.format(nodes.index(nei) + 1))
                        fNamed.write('{} 1 '.format(nei))
        f.write('\n')
        fNamed.write('\n')
    f.close()
    fNamed.close()


def readMetisResult(fileName):
    f = open(fileName, 'r')
    # print(f.name)
    result = {}
    lines = f.readlines()
    node = 1
    for line in lines:
        partion = line[:-1]
        if result.get(partion) is None:
            result[partion] = []
        result[partion].append(getLEO(node))
        node += 1
    newR = {}
    for p in result.values():
        newR[p[0]] = p
    return newR


def readMetisResultwithLink():
    f = open('metisLinkG72.part.8', 'r')
    result = {}
    lines = f.readlines()
    f1 = open('nodeOrder.txt', 'r')
    nodeOrders = eval(f1.read())
    index = 0
    for line in lines:
        partion = line[:-1]
        if result.get(partion) is None:
            result[partion] = []
        result[partion].append(nodeOrders[index])
        index += 1
    pprintDict(result)
    newR = {}
    for p in result.values():
        newR[p[0]] = [n for n in p if len(n) < 6]
        for node in p:
            if len(node) > 5:
                node1 = node.split('-')[0]
                node2 = node.split('-')[1]
                if '{}-{}'.format(node2, node1) in p:
                    if node1 not in p:
                        print('{}, {}, {} -> {}'.format(node, '{}-{}'.format(node2, node1), node2, node1))
                    if node2 not in p:
                        print('{}, {}, {} -> {}'.format(node, '{}-{}'.format(node2, node1), node1, node2))
    # pprintDict(newR)
    return newR


def getResultGraph(G, link_load):
    graph = nx.Graph()
    graph.add_nodes_from(G.nodes(data=True))
    for node in G.nodes:
        if node[-1:] == "9":
            graph.add_node('{}-1'.format(node), load=G.nodes[node]['load'], controller=G.nodes[node]['controller'])
    # print(graph.nodes(data=True))
    for edge in G.edges:
        if edge[0][-1:] == '1' and edge[1][-1:] == '9':
            graph.add_edge(edge[0], '{}-1'.format(edge[1]), delay=G.edges[edge]['delay'],
                           load='{:3.1f}'.format(link_load[(edge[0], edge[1])]))
            # graph.add_edge(edge[1],'{}-1'.format(edge[0]),delay=G.edges[edge]['delay'],load='{:3.1f}'.format(link_load[(edge[0],edge[1])]))
        else:
            graph.add_edge(edge[0], edge[1], delay=G.edges[edge]['delay'],
                           load='{:3.1f}'.format(link_load[(edge[0], edge[1])]))
    return graph


def main():
    pass


if __name__ == '__main__':
    # Distance_Dict = get_all_distance()
    files = os.listdir('topos')
    files.sort(key=lambda x: int(x.split('.')[0]))
    for f in files[:10]:
        time = int(f.split('.')[0])
        print(time)
        G = gen_topology(time)
        # sys.exit(0)
        # linkDelay = dict(sorted([(link, G.edges[link]['delay']) for link in G.edges], key=lambda x: x[1], reverse=False))
        # linkDelay = {link: G.edges[link]['delay'] for link in G.edges}
        pair_load = get_pair_load(G)
        pair_path_delay = get_pair_delay(G)
        link_load = get_link_load(G, pair_load, pair_path_delay)
        assignment = None
        if AssignScheme == 'Greedy':
            assignment = greedy_alg1(G)
        elif AssignScheme == 'SamePlane':
            assignment = assignment1
        elif AssignScheme == 'BalCon':
            # initialAssign = greedy_alg1(G)
            # initialAssign = readMetisResult()
            f = open('balconResult.txt', 'r')
            lines = f.readlines()
            if len(lines) > 1:
                mcs = lines[0].split(' ')[0]
                mssls = lines[0].split(' ')[1]
                assignment = eval(lines[2])
                f.close()
            else:
                initialAssign = assignment1
                f = open('balconResult.txt', 'w')
                assignment = bal_con_assign(initialAssign)
                f.write('{} {}\n'.format(MCS, MSSLS))
                f.write(assignment.__str__())
                f.close()
        elif AssignScheme == 'METIS':
            metisFile = 'MetisTopos/{}'.format(time)
            if not os.path.exists(metisFile):
                genMetisGraph(time)
            scheme = 'my-0'
            resultLogFile = 'metisResult1/{}/{}'.format(scheme, time)
            f = open(resultLogFile, 'w')
            f.close()
            logger = logging.getLogger('{}'.format(time))
            logger.setLevel(logging.INFO)
            f_handler = logging.FileHandler(resultLogFile)
            logger.addHandler(f_handler)
            for ufactor in range(100, 3000, 100):
                for contig in ['-contig', '']:
                    for p in range(8,9):
                    # cmd = 'gpmetis {} 8 -ufactor={} {} |tee -a {}'.format(metisFile, ufactor, contig, resultLogFile)
                        cmd = 'gpmetis {} {} -ufactor={} {} |tee -a {}'.format(metisFile, p,ufactor, contig, resultLogFile)
                        os.system(cmd)
                    # resultFile = '{}.part.8'.format(metisFile)
                        resultFile = '{}.part.{}'.format(metisFile,p)
                    # f = open('metisResult.txt', 'r')
                    # lines = f.readlines()
                    # if len(lines) > 1:
                    #     assignment = eval(lines[0])
                    #     f.close()
                    # else:
                    #     f = open('metisResult.txt', 'w')
                    #     assignment = readMetisResult()
                    #     f.write(assignment.__str__())
                    #     f.close()
                        assignment = readMetisResult(fileName=resultFile)
                        assignment,_,_ = deployInArea(G, pair_load, pair_path_delay, link_load, assignment)
                        for con in assignment:
                            # print('{}: {}'.format(con, assignment[con]))
                            logger.info('{}: {}'.format(con, assignment[con]))
                            # logger.info('ok')
                        con_loads = apply_placement_assignment(G, pair_load, pair_path_delay, assignment)
                        # g = getResultGraph(G)
                        # draw_result(g, assignment, node_style='load')
                        # draw_result(g, assignment, node_style='leo')
                        # draw_result(g, assignment, node_style='leo_weight')
                        # draw_result(g, assignment, node_style='load_weight')
                        avg_setup_t = get_avg_flow_setup_time(G, pair_load, pair_path_delay)
                        # logger.info('con_loads:', con_loads)
                        logger.info('sum_con_loads: {:>7.2f}'.format(sum(con_loads.values())))
                        logger.info('max_con_loads: {:>7.2f}'.format(max(con_loads.values())))
                        logger.info('avg_setup_time: {:>6.2f}'.format(avg_setup_t))
        elif AssignScheme == 'PyMETIS':
            graph = gen_init_graph(G, link_load)
            run_metis_main(graph)
            # sys.exit(0)
        # graph = gen_init_graph(G, link_load)
        # run_main(graph,k=8,level=0)
        # sys.exit(0)
        # print(link_load)
        # for i in range(30,40):
        # print(g.edges(data=True))
        # genMetisGraph(time)
        # genMetisGraphwithLink()
        # readMetisResultwithLink()
        # sys.exit(0)
        # link_delay = {}
        # for edge in G.edges:
        #     link_delay[edge] = G.edges[edge]['delay']
    # sortD = dict(sorted(link_delay.items(), key=lambda x: x[1], reverse=False))
    # print(sortD)
    # with open('result1.txt','w') as f:
    #     f.write('link delay\n')
    #     sortD=dict(sorted(link_delay.items(), key=lambda x: x[1], reverse=False))
    #     for k in sortD:
    #         f.write('{}: {}\n'.format(k,sortD[k]))
    # with open('result2.txt','w') as f:
    #     f.write('link loads\n')
    #     sortD = dict(sorted(link_load.items(), key=lambda x: x[1], reverse=True))
    #     for k in sortD:
    #         f.write('{}: {}\n'.format(k, sortD[k]))
    # sys.exit(0)
    '''
    assignment = None
    if AssignScheme == 'Greedy':
        assignment = greedy_alg1(G)
    elif AssignScheme == 'SamePlane':
        assignment = assignment1
    elif AssignScheme == 'BalCon':
        # initialAssign = greedy_alg1(G)
        # initialAssign = readMetisResult()
        f=open('balconResult.txt','r')
        lines=f.readlines()
        if len(lines)>1:
            mcs = lines[0].split(' ')[0]
            mssls = lines[0].split(' ')[1]
            assignment=eval(lines[2])
            f.close()
        else:
            initialAssign = assignment1
            f=open('balconResult.txt','w')
            assignment = bal_con_assign(initialAssign)
            f.write('{} {}\n'.format(MCS,MSSLS))
            f.write(assignment.__str__())
            f.close()
    elif AssignScheme == 'METIS':
        # f = open('metisResult.txt', 'r')
        # lines = f.readlines()
        # if len(lines) > 1:
        #     assignment = eval(lines[0])
        #     f.close()
        # else:
        #     f = open('metisResult.txt', 'w')
        #     assignment = readMetisResult()
        #     f.write(assignment.__str__())
        #     f.close()
        assignment = readMetisResult(fileName='MyMETISResult/metisGraph72.part.8')
    assignment = deployInArea(assignment)
    con_loads = apply_placement_assignment(G, assignment)
    g = getResultGraph(G)
    draw_result(g, assignment, node_style='load')
    # draw_result(g, assignment, node_style='leo')
    # draw_result(g, assignment, node_style='leo_weight')
    # draw_result(g, assignment, node_style='load_weight')
    avg_setup_t = get_avg_flow_setup_time(G)
    print('con_loads:', con_loads)
    print('sum_con_loads: {:>7.2f}'.format(sum(con_loads.values())))
    print('max_con_loads: {:>7.2f}'.format(max(con_loads.values())))
    print('avg_setup_time: {:>6.2f}'.format(avg_setup_t))
    # pprint(G.nodes(data=True))
    '''

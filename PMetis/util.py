import logging
import os
import time
import math
import os
import linecache

import matplotlib.pyplot as plt
import networkx as nx
from networkx import path_weight, Graph
from config import MATRIX, log


def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        log.info('函数 {} 耗时：{:>5.4f}秒'.format(f.__name__, e_time - s_time))
        return res

    return inner


def get_log_handlers(des, file):
    d = str.split(des, ',')
    handlers = []
    if 'f' in d:
        handler = logging.FileHandler(file)
        handler.terminator = ''
        handlers.append(handler)
    if 's' in d:
        handler = logging.StreamHandler()
        handler.terminator = ''
        handlers.append(handler)
    return handlers


def get_lbr(loads):
    avg = sum(loads) / len(loads)
    lbr = 0
    if avg != 0:
        lbr = 1 - sum([abs(x - avg) for x in loads]) / (4 * avg)
    return lbr, avg


def draw_result(G, assignment, node_style, title='', node_loads=None):
    if node_loads is None:
        node_loads = {}
    plt.figure(figsize=(11, 8), dpi=300)
    plt.title('{}-{}'.format(title, node_style))
    # currentAxis = plt.gca()
    # rect = patches.Rectangle((0.2, 0.6), 0.5, 0.6, linewidth=0.5, edgecolor='r', facecolor='none')
    # currentAxis.add_patch(rect)
    # pos = nx.spring_layout(G, seed=seed)
    pos = {}
    for node in G:
        if len(node) > 5:
            pos[node] = [(int(node[3]) - 5) * 0.25, 1]
        else:
            pos[node] = [(int(node[3]) - 5) * 0.25, (5 - int(node[4])) * 0.2]
    # print(pos)
    # matplotlib.axes.Axes
    # node_color = [float(G.degree(v)) for v in G]
    # labels = {n:'{}-{}'.format(n[3:],G.nodes[n]['load'])  for n in pos}
    # if node_style.find('leo') != -1:
    labels = {n: '{}'.format(n[3:]) for n in pos}
    if node_style.find('load') != -1:
        # labels = {n: '{}'.format(G.nodes[n]['load']) for n in pos}
        for node in pos:
            if node.find('-'):
                labels[node] = '{:.1f}'.format(node_loads[node.split('-')[0]])
            else:
                labels[node] = '{:.1f}'.format(node_loads[node])
    colors = {0: [0.5509678955659282, 0.03551278107210043, 0.9882117954823949],
              1: [0.03210331669684463, 0.9580241663267244, 0.45233045746584133],
              2: [0.3680757817092095, 0.8794628767330064, 0.9362373012420029],
              3: [0.9694056331323755, 0.4433425232505013, 0.06437686034548362],
              4: [0.11064767393618924, 0.5989936058369307, 0.06477158156975393],
              5: [0.5137800981359481, 0.3631809784548581, 0.15032575180474528],
              6: [0.5680757817092095, 0.6794628767330064, 0.9362373012420029],
              7: [0.78257392844631, 0.4348129490990973, 0.8054749944623513]}
    controllers = list(assignment.keys())
    con_colors = {}
    for i in range(len(controllers)):
        con_colors[controllers[i]] = colors[i]
    # print(con_colors)
    node_colors = []
    node_sizes = []
    for n in pos:
        controller = G.nodes[n]['controller']
        if controller == n:
            node_sizes.append(600)
        else:
            node_sizes.append(300)
        # print(controller)
        color = con_colors[controller]
        node_colors.append(color)
    if node_style.find('weight') != -1:
        node_colors = [[1 - int(G.nodes[n]['load']) / 7, 0.7, 0.7] for n in pos]
    # print(node_colors)

    nx.draw(G, pos, labels=labels, with_labels=True, node_color=node_colors, node_size=node_sizes)

    # nx.draw_networkx()
    # nx.drawing.layout
    # nx.draw_networkx_nodes()
    # nx.draw_networkx_edges()
    edge_weight = nx.get_edge_attributes(G, 'load')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weight)
    plt.savefig('resultFig/{}-{}.png'.format(title, node_style))
    plt.show()


def gen_topology(time_slot):
    graph = nx.Graph()
    plane_num = 8
    sat_per_plane = 9
    for p_id in range(1, 1 + plane_num):
        for s_id in range(1, sat_per_plane + 1):
            src = '{}{}'.format(p_id, s_id)
            srcLEO = 'LEO{}'.format(src)
            if not graph.has_node(srcLEO):
                graph.add_node(srcLEO, load=getLoad(src, time_slot), controller='', con_load=0, real_load=0)
            n_s_id = s_id + 1
            if n_s_id > 9:
                n_s_id %= sat_per_plane
            dst = '{}{}'.format(p_id, n_s_id)
            dstLEO = 'LEO{}'.format(dst)
            if not graph.has_node(dstLEO):
                graph.add_node(dstLEO, load=getLoad(dst, time_slot), controller='', con_load=0, real_load=0)
            # print('LEO{}'.format(src), '>', 'LEO{}'.format(dst))
            graph.add_edge(srcLEO, dstLEO, delay=16.32)
    connections = open('topos/{}.log'.format(time_slot), 'r').readlines()
    for connect in connections:
        src = connect[:2]
        dst = connect[3:5]
        # print('LEO{}'.format(src), '>', 'LEO{}'.format(dst))
        graph.add_edge('LEO{}'.format(src), 'LEO{}'.format(dst), delay=get_delay(src, dst, time_slot))
    return graph


def getIndex(string):
    i = int(string[3])
    j = int(string[4])
    return (i - 1) * 10 + j - i + 1


def getLEO(index):
    if index % 9 == 0:
        return 'LEO{:.0f}9'.format(math.ceil(index / 9))
    else:
        return 'LEO{:.0f}{}'.format(math.ceil(index / 9), index % 9)


def getIndexWithL(string):
    i = int(string[3])
    j = int(string[4])
    return (i - 1) * 10 + j - i + 1


def getLEOWithL(index):
    if index % 9 == 0:
        return 'LEO{:.0f}9'.format(math.ceil(index / 9))
    else:
        return 'LEO{:.0f}{}'.format(math.ceil(index / 9), index % 9)


def get_delay(src, dst, time):
    f = open('../data/72-d/{}-{}.csv'.format(src, dst), 'r')
    line = f.readlines()[int(time / 10) + 1].split(',')
    return float(line[1]) / (3 * 10 ** 5)


def get_all_load(graph: Graph):
    all_load = 0
    for node in graph.nodes:
        all_load += graph.nodes[node]['load']
    return all_load


def get_all_distance():
    all_dis_dict = {}
    files = os.listdir('../72-distance/')
    for file in files:
        key = file.split('.')[0]
        all_dis_dict[key] = []
        lines = open('../72-distance/{}'.format(file), 'r').readlines()[1:]
        for line in lines:
            all_dis_dict[key].append(line.split(','))
    return all_dis_dict


def get_pair_load(graph):
    """
    获取所有两个卫星间的负载

    :param graph: 卫星图
    :return: 所有两两卫星间的负载
    """
    all_pairs_load = {}
    all_load = get_all_load(graph)
    for src in graph.nodes:
        src_load = graph.nodes[src]['load']
        for dst in graph.nodes:
            if src != dst:
                dst_load = graph.nodes[dst]['load']
                all_pairs_load[(src, dst)] = src_load * (dst_load / (all_load - src_load))
            else:
                all_pairs_load[(src, dst)] = 0
    return all_pairs_load


def get_pair_delay(graph):
    """
    获取端到端路径时延
    :param graph: 网络拓扑图
    :return: 网络内所有端到端的路径及对应时延
    """
    all_src_dst_paths = {}
    for src in graph.nodes:
        for dst in graph.nodes:
            # all_src_dst_paths[(src, dst)] = []
            if src != dst:
                path = nx.shortest_path(graph, src, dst, weight='delay')
                all_src_dst_paths[(src, dst)] = [path, path_weight(graph, path, 'delay')]
                # print('{}->{}:'.format(src, dst))
                # paths = list(islice(nx.shortest_simple_paths(graph, src, dst, weight='delay'), 4))
                # if len(paths) > 1:
                #     for path in paths:
                #         print('{}:{}'.format(path, path_weight(graph, path, 'delay')))
                #         all_src_dst_paths[(src, dst)].append([path, path_weight(graph, path, 'delay')])
                # else:
                #     print('{}'.format(paths))
            else:
                all_src_dst_paths[(src, dst)] = [[src], 0]
    return all_src_dst_paths


def new_file(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    f = open(path, 'w')
    f.close()


def get_edge_load(graph, pair_load, pair_path_delay):
    """
    获取图中每条边上通过的负载, 双向的

    :param graph: graph
    :param pair_load:
    :param pair_path_delay:
    :return: each edge load
    """
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


class Common():

    def __init__(self, graph: Graph):
        self.graph = graph
        self.pair_load = get_pair_load(graph)
        self.pair_path_delay = get_pair_delay(graph)
        self.link_load = get_edge_load(graph, self.pair_load, self.pair_path_delay)


class Ctrl():
    def __init__(self):
        self.coarsenTo = 240
        self.nIparts = 4
        self.nCuts = 4
        self.niter = 10
        self.seed=0
        self.ubfactors=1.13
        self.max_v_wgt=0


def exch(dicts):
    new_dicts = {}
    for k in dicts:
        new_dicts[str(k)] = dicts[k]
    return new_dicts


def pprintDict(map):
    print('{')
    for k in map:
        print('{}: {}'.format(k, map[k]))
    print('}')


def genEachTTopo():
    changeFile = open('allTopoChange.csv', 'r')
    initFile = open('topos/0.log', 'r')
    edges = initFile.readlines()
    changeLines = changeFile.readlines()
    lastTime = 0
    for line in changeLines:
        lineList = line.split(',')
        time = float(lineList[0])
        if abs(time - lastTime) > 1 and lastTime != 0:
            newTFile = open('topos/{}.log'.format(int(lastTime)), 'w')
            for e in edges:
                newTFile.write(e)
            newTFile.close()
        if lineList[1] == 'up':
            edges.append(lineList[2])
        else:
            edges.remove(lineList[2])
        lastTime = time


def position_to_index(lat, lon):
    """
    根据经纬度，将卫星映射到矩阵内
    :param lat: 卫星纬度
    :param lon: 卫星经度
    :return: 卫星在矩阵的index
    """
    x = math.floor((-lat + 90) / 15)
    if lon >= 0:
        y = math.floor(lon / 15)
    else:
        y = 24 - math.ceil((-lon) / 15)
    return x, y


def getLoad(leo, time):
    f_name = '../data/72-loads/srcData/{}.csv'.format(leo)
    f = open(f_name, 'r')
    # lines = f.readlines()
    # line = f.readlines()[time + 1].split(',')
    line = linecache.getline(f_name, time + 2).split(',')
    # line = lines[1].split(',')
    # time = float(line[0])
    lat = float(line[1]) * 180 / math.pi
    lon = float(line[2]) * 180 / math.pi
    x, y = position_to_index(lat, lon)
    load = MATRIX[x][y]
    # print(leo,time, load)
    # return x, y
    return load


def getLoadChangeFile():
    files = os.listdir('../72-loads/')[:-1]
    print(files)
    for fileN in files:
        file = open('../72-loads/' + fileN, 'r')
        lastLoad = -1
        lines = file.readlines()
        changeFile = open('../72-loads/change/' + fileN, 'w')
        for line in lines[1:]:
            line = line.split(',')
            time = float(line[0])
            lat = float(line[1]) * 180 / math.pi
            lon = float(line[2]) * 180 / math.pi
            x, y = position_to_index(lat, lon)
            load = MATRIX[x][y]
            if lastLoad != load:
                changeFile.write('{},{},{},{}\n'.format(time, lat, lon, load))
            lastLoad = load


def getAllLoadChange():
    # allChangeFile = open('allLoadChange.csv', 'w')
    # files = os.listdir('../72-loads/change/')
    allChangeFile = open('Topo-Load-Change.csv', 'w')
    files = ['allLoadChange.csv', 'allTopoChange.csv']
    print(files)
    all_file_dict = []
    for fileN in files:
        # file = open('../72-loads/change/' + fileN, 'r')
        file = open(fileN, 'r')
        LEO_name = fileN.split('.')[0]
        file_dict = {LEO_name: []}
        lines = file.readlines()
        for line in lines:
            # line = line.split(',')
            # t = float(line[0])
            # load = float(line[-1])
            file_dict[LEO_name].append(line.split(','))
        all_file_dict.append(file_dict)
    while len(all_file_dict) > 0:
        min_time = math.inf
        file_index = 0
        min_index = 0
        key = ''
        min_key = ''
        while file_index < len(all_file_dict):
            key = list(all_file_dict[file_index].keys())[0]
            if float(all_file_dict[file_index][key][0][0]) < min_time:
                min_time = float(all_file_dict[file_index][key][0][0])
                min_index = file_index
                min_key = key
            file_index += 1
        allChangeFile.write(
            '{},{},{}'.format(all_file_dict[min_index][min_key][0][0], all_file_dict[min_index][min_key][0][1],
                              all_file_dict[min_index][min_key][0][2]))
        all_file_dict[min_index][min_key].pop(0)
        if len(all_file_dict[min_index][min_key]) == 0:
            all_file_dict.pop(min_index)

def rename_sum_file():
    import pathlib
    files=list(pathlib.Path('./metis/').glob('**/*.sum'))
    print(len(files))
    for f in files:
        s_name=f.name
        t_name=f.parent.absolute().__str__()+'/-{}-sum.json'.format(s_name.split('.')[0])
        print('{}->{}'.format(s_name,t_name))
        f.rename(t_name)
        

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
        sum_val += graph.nodes[node]['load']
        neighbors = nx.neighbors(graph, node)
        for nei in neighbors:
            nei_partition = graph.nodes[nei]['belong']
            if node_p != nei_partition:
                cut += graph.edges[(node, nei)]['wei']
                node_ed[node] += graph.edges[(node, nei)]['wei']
            else:
                node_id[node] += graph.edges[(node, nei)]['wei']
        if node_ed[node] - node_id[node]> 0 :
            boundary_nodes.append(node)
            # print("before add boundary node {}".format(node))
    graph.graph['sum_val'] = sum_val
    graph.graph['boundary'] = boundary_nodes
    graph.graph['cut'] = cut / 2
    graph.graph['p_vals'] = part_val
    graph.graph['node_id'] = node_id
    graph.graph['node_ed'] = node_ed
    graph.graph['part'] = part
    return part, part_val



if __name__ == '__main__':
    # rename_sum_file()
    absP = os.path.abspath('../matlab_Code/72-d')
    files = os.listdir('../matlab_Code/72-d')
    for f in files:
        newF = f[4:]
        os.rename(os.path.join(absP, f), os.path.join(absP, newF))

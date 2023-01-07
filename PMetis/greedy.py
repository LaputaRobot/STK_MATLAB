import copy

import networkx as nx

from analysis_result import apply_partition
from util import Common


def greedy_alg1(common:Common):
    """
    以贪心算法, 按边权重结合节点, 效果不行

    :param graph:
    :return:
    """
    # con_cap = 80
    graph=common.graph
    link_load=common.link_load
    pair_load = common.pair_load
    pair_path_delay = common.pair_path_delay
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
    apply_partition(copy_graph,common.link_load, result)
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
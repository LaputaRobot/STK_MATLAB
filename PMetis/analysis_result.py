from PMetis.util import *


def apply_partition(graph, link_load, partition):
    """
    将划分结果应用于各交换机, 并计算各控制器的控制器负载

    :param link_load:
    :param graph: graph
    :param partition: partition result
    :return: each controller load
    """
    con_loads = {}
    for p in partition:
        con_load = 0
        for sw in partition[p]:
            graph.nodes[sw]['controller'] = p
            graph.nodes[sw]['real_load'] = graph.nodes[sw]['load']
            neighbors = nx.neighbors(graph, sw)
            for nei in neighbors:
                if nei not in partition[p]:
                    graph.nodes[sw]['real_load'] += link_load[(nei, sw)]
            con_load += graph.nodes[sw]['real_load']
        con_loads[p] = con_load
        for sw in partition[p]:
            graph.nodes[sw]['con_load'] = con_load
    return con_loads


def get_avg_flow_setup_time(graph, pair_load, pair_path_delay):
    '''
    获取所有流量对安装时延的加权平均值

    :param graph: 图
    :param pair_load: peer-peer load
    :param pair_path_delay:  peer-peer path and delay
    :return: 流安装时延的平均值
    '''
    all_pairs_setup_time = 0
    for pair in pair_load:
        if pair_load[pair] != 0:
            setup_time = pair_path_delay[pair][1]
            path = pair_path_delay[pair][0]
            same_con_switches = []
            for node_index in range(len(path)):
                sw = path[node_index]
                if node_index == 0 or graph.nodes[sw]['controller'] != graph.nodes[path[node_index - 1]]['controller']:
                    propagation_delay = pair_path_delay[(sw, graph.nodes[sw]['controller'])][1]
                    con_process_delay = max(0, 1 / (100 - graph.nodes[path[node_index]]['con_load']) * 1000)
                    setup_time += (propagation_delay + con_process_delay)
                    same_con_switches = [sw]
                if node_index == len(path) - 1 or graph.nodes[sw]['controller'] != graph.nodes[path[node_index + 1]][
                    'controller']:
                    install_rule_time = max(
                        [pair_path_delay[(sw, graph.nodes[sw]['controller'])][1] for sw in same_con_switches])
                    setup_time += install_rule_time
                    same_con_switches = []
                else:
                    same_con_switches.append(sw)
            all_pairs_setup_time += setup_time * pair_load[pair]
    return all_pairs_setup_time / sum(pair_load.values())


def deploy_in_area(graph, pair_path_delay, link_load, assign):
    apply_partition(graph, link_load, assign)
    new_result = {}
    sum_min_delay = 0
    for con in assign:
        area = assign[con]
        min_delay_con = None
        min_delay = 10000
        for newCon in area:
            weighted_delay = 0
            for sw in area:
                if newCon != sw:
                    weighted_delay += graph.nodes[sw]['controller'] * pair_path_delay[(sw, newCon)][1]
            if weighted_delay < min_delay:
                min_delay = weighted_delay
                min_delay_con = newCon
        # print('{} all delay: {}'.format(min_delay_con,min_delay))
        sum_min_delay += min_delay
        new_result[min_delay_con] = area
    # print('*'*20,'node loads','*'*20, '\n',exch(node_loads))
    # print('sum_min_delay {}'.format(sum_min_delay))
    # logger.info('*' * 20, 'Final Assignment', '*' * 20)
    # return new_result
    return new_result, sum_min_delay


def get_result_graph(G, link_load):
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

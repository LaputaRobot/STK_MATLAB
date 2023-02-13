from util import *

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
                    weighted_delay += graph.nodes[sw]['real_load'] * pair_path_delay[(sw, newCon)][1]
            if weighted_delay < min_delay:
                min_delay = weighted_delay
                min_delay_con = newCon
        # print('{} all delay: {}'.format(min_delay_con,min_delay))
        sum_min_delay += min_delay
        new_result[min_delay_con] = area
    return new_result, sum_min_delay

def get_avg_flow_setup_time(graph, pair_load, pair_path_delay):
    '''
    获取所有流量对安装时延的加权平均值

    :param graph: 图
    :param pair_load: peer-peer load
    :param pair_path_delay:  peer-peer path and delay
    :return: 流安装时延的平均值
    '''
    all_pairs_setup_time = 0
    nodes= graph.nodes()
    sum_load= 0
    for pair in pair_load:
        if pair_load[pair] != 0:
            setup_time = pair_path_delay[pair][1] # 路由时延
            path = pair_path_delay[pair][0]
            install_rule_time = 0
            path_len =len(path)
            for node_index in range(path_len):
                sw = path[node_index]
                con = nodes[sw]['controller']
                propagation_delay = pair_path_delay[(sw, con)][1]
                install_rule_time = max(install_rule_time, propagation_delay)
                if node_index == 0 or con != nodes[path[node_index - 1]]['controller']:
                    con_process_delay = 1 / (100 - nodes[sw]['con_load']) * 1000
                    setup_time += (propagation_delay + con_process_delay)
                if node_index == path_len - 1 or con != nodes[path[node_index + 1]][
                    'controller']:
                    setup_time += install_rule_time
                    install_rule_time = 0
            all_pairs_setup_time += setup_time * pair_load[pair]
            sum_load += pair_load[pair]
    return all_pairs_setup_time / sum_load


def analysis(common: Common, assignment, logger):
    assignment, con_loads = deploy_in_area(common.graph, common.pair_path_delay, common.link_load, assignment)
    con_loads = apply_partition(common.graph, common.link_load, assignment)
    avg_setup_t = get_avg_flow_setup_time(common.graph, common.pair_load, common.pair_path_delay)
    for con in assignment:
        logger.info('{}: {}: {}\n'.format(con, con_loads[con], assignment[con]))
    logger.info('sum_con_loads: {:>7.2f}\n'.format(sum(con_loads.values())))
    logger.info('max_con_loads: {:>7.2f}\n'.format(max(con_loads.values())))
    logger.info('avg_setup_time: {:>6.2f}\n'.format(avg_setup_t))


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


def draw_result_with_time(scheme, T, node_style):
    G = gen_topology(T)
    pair_load = get_pair_load(G)
    pair_path_delay = get_pair_delay(G)
    link_load = get_edge_load(G, pair_load, pair_path_delay)
    assignment = {}
    src_args, ass, _, _, _ = get_min_delay('metisResult1/{}/{}'.format(scheme, T))
    for a in ass:
        assignment[a.split(':')[0]] = eval(a.split(':')[1])
    assignment, sum_min_delay = deploy_in_area(G, pair_path_delay, link_load, assignment)
    print('{}-{}'.format(T, scheme))
    for con in assignment:
        # print('{}: {}'.format(con, assignment[con]))
        print('{}: {}'.format(con, assignment[con]))
    con_loads = apply_partition(G, link_load, assignment)
    graph = get_result_graph(G, link_load)
    print(con_loads)
    # g = getResultGraph()
    # draw_result(g, assignment, node_style='load')
    # draw_result(g, assignment, node_style='leo')
    # draw_result(g, assignment, node_style='leo_weight')
    # draw_result(g, assignment, node_style='load_weight')
    pair_load = dict(sorted(pair_load.items(), key=lambda x: x[1], reverse=True))
    avg_setup_t = get_avg_flow_setup_time(G, pair_load, pair_path_delay)
    # logger.info('con_loads:', con_loads)
    print('sum_con_loads: {:>7.2f}'.format(sum(con_loads.values())))
    print('max_con_loads: {:>7.2f}'.format(max(con_loads.values())))
    print('avg_setup_time: {:>6.2f}'.format(avg_setup_t))
    print('avg_setup_time: {:>6.2f}'.format(sum_min_delay / sum(con_loads.values())))
    print('sum_min_delay: {:>6.2f}'.format(sum_min_delay))

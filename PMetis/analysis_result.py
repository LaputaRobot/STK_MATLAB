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
    assignment, sum_min_delay = deploy_in_area(common.graph, common.pair_path_delay, common.link_load, assignment)
    con_loads = apply_partition(common.graph, common.link_load, assignment)
    avg_setup_t = get_avg_flow_setup_time(common.graph, common.pair_load, common.pair_path_delay)
    for con in assignment:
        logger.info('{}: {}: {}\n'.format(con, con_loads[con], assignment[con]))
    logger.info('sum_con_loads: {:>7.2f}\n'.format(sum(con_loads.values())))
    logger.info('max_con_loads: {:>7.2f}\n'.format(max(con_loads.values())))
    logger.info('avg_setup_time: {:>6.2f}\n'.format(avg_setup_t))
    logger.info('sum_min_delay: {:>6.2f}\n'.format(sum_min_delay))



def get_result_graph(G, link_load, assignment):
    graph = nx.Graph()
    graph.add_nodes_from(G.nodes(data=True))
    for con in assignment:
        for node in assignment[con]:
            graph.nodes[node]['controller']= con
            if node[-1:] == "9":
                graph.add_node('{}-v'.format(node), load=G.nodes[node]['load'], controller=graph.nodes[node]['controller'])
    for edge in G.edges:
        if edge[0][-1:] == '1' and edge[1][-1:] == '9':
            graph.add_edge(edge[0], '{}-v'.format(edge[1]), delay=G.edges[edge]['delay'],
                           load='{:3.1f}'.format(link_load[(edge[0], edge[1])]))
        else:
            graph.add_edge(edge[0], edge[1], delay=G.edges[edge]['delay'],
                           load='{:3.1f}'.format(link_load[(edge[0], edge[1])]))
    return graph


def draw(graph, assignment, figure_path):
    plt.figure(figsize=(11, 8), dpi=300)
    pos = {}
    for node in graph:
        if len(node) > 5:
            pos[node] = [(int(node[3]) - 5) * 0.25, 1]
        else:
            pos[node] = [(int(node[3]) - 5) * 0.25, (5 - int(node[4])) * 0.2]
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
    labels = {}
    node_colors = []
    node_sizes = []   
    for node in pos:
        labels[node] = '{}'.format(graph.nodes[node]['load'])
        controller = graph.nodes[node]['controller']
        if controller == node:
            node_sizes.append(600)
        else:
            node_sizes.append(300)
        node_colors.append(con_colors[controller])
    nx.draw(graph, pos, labels=labels, with_labels=True,
            node_color=node_colors, node_size=node_sizes)
    edge_weight = nx.get_edge_attributes(graph, 'load')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weight)
    new_file(figure_path)
    plt.savefig(figure_path)
    plt.show()


def draw_log_result(t, scheme, log_file):
    figure_dir = os.path.join(result_base,'part/figure/{}/{}'.format(scheme, t))
    figure_path = os.path.join(figure_dir,'{}.png'.format(log_file.split('.')[0]))
    log_path = ''
    if scheme =='metis':
        log_path = os.path.join(result_base, 'part/{}/log/{}/{}'.format(scheme,t, log_file))
    if scheme == 'pymetis':
        log_path = os.path.join(result_base, 'part/{}/{}/{}'.format(scheme,t, log_file))
    f = open(log_path, 'r')
    lines = f.readlines()
    ass = {}
    for line in lines:
        if 'LEO' in line and '[' in line:
            cols = line.split(':')
            con = cols[0]
            switches = eval(cols[2])
            ass[con] = switches
    common=Common(t)
    link_load= common.link_load
    g = common.graph
    graph=get_result_graph(g, link_load,ass)
    draw(graph, ass, figure_path)

if __name__ == "__main__":
    # draw_log_result(0, 'metis', '8-700-3-contig.log')
    draw_log_result(93, 'pymetis', '8-1100-Wei-WeiLoad-0-contig.log')
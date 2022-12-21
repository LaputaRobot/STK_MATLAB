


def get_avg_flow_setup_time(graph, pair_load, pair_path_delay):
    '''
    获取所有流量对的安装时延的平均值

    :param graph: 图
    :param pair_load: 流量对
    :param pair_path_delay: 节点间的时延
    :return: 流安装时延的平均值
    '''
    all_pairs_setup_time = 0
    len_sum = 0
    len_num = 0
    for pair in pair_load:
        if pair_load[pair] != 0:
            # print('get pair {}'.format(pair))
            setup_time = pair_path_delay[pair][1]
            path = pair_path_delay[pair][0]
            len_sum += len(path)
            len_num += 1
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


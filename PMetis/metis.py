import copy

from PMetis.util import Common, getIndex, getLEO, pprintDict


def gen_metis_file(time,common:Common):
    """
    基于时间戳生成用于metis程序输入的文件

    :param time: 时间戳
    """
    f = open('MetisTopos/{}'.format(time), 'w')
    G=common.graph
    link_load=common.link_load
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



def read_metis_result(file_name):
    '''
    获取metis划分结果

    :param file_name: 划分结果文件名
    :return:
    '''
    f = open(file_name, 'r')
    # print(f.name)
    result = {}
    lines = f.readlines()
    node = 1
    for line in lines:
        partition = line[:-1]
        if result.get(partition) is None:
            result[partition] = []
        result[partition].append(getLEO(node))
        node += 1
    newR = {}
    for p in result.values():
        newR[p[0]] = p
    return newR

def genMetisGraphwithLink(common:Common):
    G=common.graph
    link_load=common.link_load
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
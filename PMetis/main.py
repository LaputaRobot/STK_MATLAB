import logging
import os

from PMetis.PyMetis import run_metis_main, gen_init_graph
from PMetis.analysis_result import deploy_in_area, apply_partition, get_avg_flow_setup_time
from PMetis.balcon import bal_con_assign
from PMetis.config import AssignScheme, assignment1, MCS, MSSLS
from PMetis.greedy import greedy_alg1
from PMetis.metis import gen_metis_file, read_metis_result
from PMetis.util import gen_topology, Common

if __name__ == '__main__':
    # Distance_Dict = get_all_distance()
    files = os.listdir('topos')
    files.sort(key=lambda x: int(x.split('.')[0]))
    for f in files[:10]:
        t = int(f.split('.')[0])
        print(t)
        G = gen_topology(t)
        common=Common(G)
        # sys.exit(0)
        # linkDelay = dict(sorted([(link, G.edges[link]['delay']) for link in G.edges], key=lambda x: x[1], reverse=False))
        # linkDelay = {link: G.edges[link]['delay'] for link in G.edges}
        # pair_load = get_pair_load(G)
        # pair_path_delay = get_pair_delay(G)
        # link_load = get_edge_load(G, pair_load, pair_path_delay)

        assignment = None
        if AssignScheme == 'Greedy':
            assignment = greedy_alg1(common)
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
                assignment = bal_con_assign(common,initialAssign)
                f.write('{} {}\n'.format(MCS, MSSLS))
                f.write(assignment.__str__())
                f.close()
        elif AssignScheme == 'METIS':
            metisFile = 'MetisTopos/{}'.format(t)
            if not os.path.exists(metisFile):
                gen_metis_file(t)
            scheme = 'my-0'
            resultLogFile = 'metisResult1/{}/{}'.format(scheme, t)
            f = open(resultLogFile, 'w')
            f.close()
            logger = logging.getLogger('{}'.format(t))
            logger.setLevel(logging.INFO)
            f_handler = logging.FileHandler(resultLogFile)
            logger.addHandler(f_handler)
            for ufactor in range(100, 3000, 100):
                for contig in ['-contig', '']:
                    for p in range(8, 9):
                        cmd = 'gpmetis {} {} -ufactor={} {} |tee -a {}'.format(metisFile, p, ufactor, contig,
                                                                               resultLogFile)
                        os.system(cmd)
                        # resultFile = '{}.part.8'.format(metisFile)
                        resultFile = '{}.part.{}'.format(metisFile, p)
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
                        assignment = read_metis_result(file_name=resultFile)
                        assignment, _, _ = deploy_in_area(G, common.pair_load, common.link_load, assignment)
                        for con in assignment:
                            # print('{}: {}'.format(con, assignment[con]))
                            logger.info('{}: {}'.format(con, assignment[con]))
                            # logger.info('ok')
                        con_loads = apply_partition(common.graph,common.link_load, assignment)
                        # g = getResultGraph(G)
                        # draw_result(g, assignment, node_style='load')
                        # draw_result(g, assignment, node_style='leo')
                        # draw_result(g, assignment, node_style='leo_weight')
                        # draw_result(g, assignment, node_style='load_weight')
                        avg_setup_t = get_avg_flow_setup_time(common.graph, common.pair_load, common.pair_path_delay)
                        # logger.info('con_loads:', con_loads)
                        logger.info('sum_con_loads: {:>7.2f}'.format(sum(con_loads.values())))
                        logger.info('max_con_loads: {:>7.2f}'.format(max(con_loads.values())))
                        logger.info('avg_setup_time: {:>6.2f}'.format(avg_setup_t))
        elif AssignScheme == 'PyMETIS':
            graph = gen_init_graph(G, common.link_load)
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
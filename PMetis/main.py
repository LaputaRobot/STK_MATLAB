import logging
import os
import shutil

from PMetis.PyMetis import run_metis_main, gen_init_graph
from PMetis.analysis_result import deploy_in_area, apply_partition, get_avg_flow_setup_time, analysis
from PMetis.balcon import bal_con_assign
from PMetis.config import AssignScheme, assignment1, MCS, MSSLS, Rewrite, assignment2, LogDestination
from PMetis.greedy import greedy_alg1
from PMetis.metis import gen_metis_file, read_metis_result
from PMetis.util import gen_topology, Common, new_file, get_log_handlers

if __name__ == '__main__':
    files = os.listdir('topos')
    files.sort(key=lambda x: int(x.split('.')[0]))
    for f in files[:1]:
        t = int(f.split('.')[0])
        print(t)
        G = gen_topology(t)
        common = Common(G)
        assignment = None
        if AssignScheme == 'Greedy':
            assignment = greedy_alg1(common)
        elif AssignScheme == 'SamePlane':
            assignment = assignment1
        elif AssignScheme == 'BalCon':
            bal_assign_f = 'balcon/{}/{}-{}.ass'.format(t, MCS, MSSLS)
            if not os.path.exists(bal_assign_f) or Rewrite:
                bal_log_f = 'balcon/{}/{}-{}.log'.format(t, MCS, MSSLS)
                new_file(bal_log_f)
                logger = logging.getLogger('{}'.format(t))
                logger.setLevel(logging.INFO)
                handlers = get_log_handlers(LogDestination, bal_log_f)
                for handler in handlers:
                    logger.addHandler(handler)
                initialAssign = assignment2
                f = open(bal_assign_f, 'w')
                assignment = bal_con_assign(common, initialAssign, logger)
                f.write(assignment.__str__())
                f.close()
                analysis(common, assignment, logger)
        elif AssignScheme == 'METIS':
            metisFile = 'MetisTopos/{}'.format(t)
            if not os.path.exists(metisFile):
                gen_metis_file(t, common)
            scheme = 'src'
            for p in range(8, 9):
                for ufactor in range(300, 3000, 100):
                    for contig in ['-contig', '']:
                        assignmentFile = 'metis/{}/{}/{}-{}{}.ass'.format(scheme, t, p, ufactor, contig)
                        if os.path.exists(assignmentFile) and not Rewrite:
                            continue
                        resultLogFile = 'metis/{}/{}/{}-{}{}.log'.format(scheme, t, p, ufactor, contig)
                        new_file(resultLogFile)
                        logger = logging.getLogger('{}-{}-{}-{}'.format(t, p, ufactor, contig))
                        logger.setLevel(logging.INFO)
                        handlers = get_log_handlers(LogDestination, resultLogFile)
                        for handler in handlers:
                            logger.addHandler(handler)
                        # cmd = 'gpmetis {} {} -ufactor={} {} |tee -a {}'.format(metisFile, p, ufactor, contig,
                        #                                                        resultLogFile)
                        cmd = 'gpmetis {} {} -ufactor={} {} > {}'.format(metisFile, p, ufactor, contig,
                                                                               resultLogFile)
                        resultFile = '{}.part.{}'.format(metisFile, p)
                        os.system(cmd)
                        shutil.move(resultFile, assignmentFile)
                        assignment = read_metis_result(file_name=assignmentFile)
                        analysis(common, assignment, logger)
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

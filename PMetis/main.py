import logging
import os
import shutil

from PyMetis import run_metis_main
from analysis_result import deploy_in_area, apply_partition, get_avg_flow_setup_time, analysis
from balcon import bal_con_assign
from config import AssignScheme, assignment1, MCS, MSSLS, Rewrite, assignment2, LogDestination
from greedy import greedy_alg1
from metis import gen_metis_file, read_metis_result
from util import gen_topology, Common, new_file, get_log_handlers

if __name__ == '__main__':
    files = os.listdir('topos')
    files.sort(key=lambda x: int(x.split('.')[0]))
    t_index = 0
    for f in files[:1]:
        t = int(f.split('.')[0])
        t_index += 1
        print("\n时隙: {}, {}".format(t_index, t))
        assignment = None
        if AssignScheme == 'Greedy':
            G = gen_topology(t)
            common = Common(G)
            assignment = greedy_alg1(common)
        elif AssignScheme == 'SamePlane':
            assignment = assignment1
        elif AssignScheme == 'BalCon':
            bal_assign_f = 'balcon/{}/{}-{}.ass'.format(t, MCS, MSSLS)
            if not os.path.exists(bal_assign_f) or Rewrite:
                G = gen_topology(t)
                common = Common(G)
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
                analysis(common, assignment, logger)
                f.write(assignment.__str__())
                f.close()
        elif AssignScheme == 'METIS':
            metisFile = 'MetisTopos/{}'.format(t)
            common = None
            if not os.path.exists(metisFile):
                G = gen_topology(t)
                common = Common(G)
                gen_metis_file(t, common)
            scheme = 'src'
            for p in range(8, 9):
                for ufactor in range(100, 3000, 100):
                    for contig in ['-contig', '']:
                        assignmentFile = 'metis/{}/{}/{}-{}{}.ass'.format(
                            scheme, t, p, ufactor, contig)
                        if os.path.exists(assignmentFile) and not Rewrite:
                            continue
                        if common is None:
                            G = gen_topology(t)
                            common = Common(G)
                        resultLogFile = 'metis/{}/{}/{}-{}{}.log'.format(
                            scheme, t, p, ufactor, contig)
                        new_file(resultLogFile)
                        logger = logging.getLogger(
                            '{}-{}-{}-{}'.format(t, p, ufactor, contig))
                        logger.setLevel(logging.INFO)
                        handlers = get_log_handlers(
                            LogDestination, resultLogFile)
                        for handler in handlers:
                            logger.addHandler(handler)
                        # cmd = 'gpmetis {} {} -ufactor={} {} |tee -a {}'.format(metisFile, p, ufactor, contig,
                        #                                                        resultLogFile)
                        cmd = 'gpmetis {} {} -ufactor={} {} > {}'.format(metisFile, p, ufactor, contig,
                                                                         resultLogFile)
                        resultFile = '{}.part.{}'.format(metisFile, p)
                        os.system(cmd)
                        shutil.move(resultFile, assignmentFile)
                        assignment = read_metis_result(
                            file_name=assignmentFile)
                        analysis(common, assignment, logger)
        elif AssignScheme == 'PyMETIS':
            G = gen_topology(t)
            common = Common(G)
            run_metis_main(common)

            # sys.exit(0)

import logging
import os
import shutil
import sys
import time
import multiprocessing

from PyMetis import run_metis_main,run_py_metis_con
from analysis_result import deploy_in_area, apply_partition, get_avg_flow_setup_time, analysis
from balcon import bal_con_assign
from config import *
from greedy import greedy_alg1
from metis import gen_metis_file, read_metis_result
from concurrent.futures import ThreadPoolExecutor,as_completed
from util import gen_topology, Common, new_file, get_log_handlers, Ctrl 

if __name__ == '__main__':
    dirname, filename = os.path.split(os.path.abspath(__file__))
    files = os.listdir(os.path.join(dirname,'topos'))
    files.sort(key=lambda x: int(x.split('.')[0]))
    t_index = 0
    # executor = ThreadPoolExecutor(max_workers=10)
    # tasks=[]
    pool=multiprocessing.Pool(processes=8)
    for f in files[:50]:
        t = int(f.split('.')[0])
        t_index += 1
        print('\n\033[1;36m','+'*77,"时隙: {}, {}".format(t_index, t),'+'*77,'\033[0m')
        assignment = None
        if AssignScheme == 'Greedy':
            common = Common(t)
            assignment = greedy_alg1(common)
        elif AssignScheme == 'SamePlane':
            assignment = assignment1
        elif AssignScheme == 'BalCon':
            bal_assign_f = 'balcon/{}/{}-{}.ass'.format(t, MCS, MSSLS)
            if not os.path.exists(bal_assign_f) or Rewrite:
                common = Common(t)
                bal_log_f = 'balcon/{}/{}-{}.log'.format(t, MCS, MSSLS)
                new_file(bal_log_f)
                logger = logging.getLogger('{}'.format(t))
                logger.setLevel(logging.INFO)
                handlers = get_log_handlers(LogDestination, bal_log_f)
                for handler in handlers:
                    logger.addHandler(handler)
                initialAssign = assignment2
                f = open(bal_assign_f, 'w')
                assignment = bal_con_assign(common, initialAssign,MCS,MSSLS, logger)
                analysis(common, assignment, logger)
                f.write(assignment.__str__())
                f.close()
        elif AssignScheme == 'METIS':
            metisFile = 'MetisTopos/{}'.format(t)
            common = None
            if not os.path.exists(metisFile):
                common = Common(t)
                gen_metis_file(common)
            scheme = 'src'
            for p in range(8, 9):
                for ufactor in range(100, 3000, 100):
                    for contig in ['-contig', '']:
                        for minconn in ['-minconn', '']:
                            assignmentFile = 'metis/{}/{}/{}-{}{}{}.ass'.format(
                                scheme, t, p, ufactor, contig, minconn)
                            if os.path.exists(assignmentFile) and not Rewrite:
                                continue
                            if common is None:
                                common = Common(t)
                            resultLogFile = 'metis/{}/{}/{}-{}{}{}.log'.format(
                                scheme, t, p, ufactor, contig, minconn)
                            new_file(resultLogFile)
                            logger = logging.getLogger(
                                '{}-{}-{}-{}'.format(t, p, ufactor, contig))
                            logger.setLevel(logging.INFO)
                            handlers = get_log_handlers([LogToFile,LogToScreen], resultLogFile)
                            for handler in handlers:
                                logger.addHandler(handler)
                            # cmd = 'gpmetis {} {} -ufactor={} {} |tee -a {}'.format(metisFile, p, ufactor, contig,
                            #                                                        resultLogFile)
                            cmd = 'gpmetis {} {} -ufactor={} {} {} > {}'.format(metisFile, p, ufactor, contig, minconn,
                                                                                resultLogFile)
                            resultFile = '{}.part.{}'.format(metisFile, p)
                            os.system(cmd)
                            shutil.move(resultFile, assignmentFile)
                            assignment = read_metis_result(
                                file_name=assignmentFile)
                            analysis(common, assignment, logger)
        elif AssignScheme == 'PyMetis':
            common = Common(t)
            for p in range(8, 9):
                for ufactor in range(100, 3000, 100):
                    for match_order in [MO_LoadWeiLoad,MO_SumWei,MO_Wei,MO_SRC]:
                        for match_scheme in [MS_SRC,MS_WeiDif,MS_WeiLoad]:
                            for contig in ['-contig','']:
                                resultLogFile = 'pymetis/{}/{}-{}-{}-{}{}.log'.format(
                                     t, p, ufactor,match_order, match_scheme, contig)
                                if os.path.exists(resultLogFile) and not Rewrite:
                                    continue
                                new_file(resultLogFile)
                                
                                ctrl=Ctrl()
                                ctrl.contiguous = True if contig == '-contig' else False
                                ctrl.nparts = p
                                ctrl.un_factor =1+ ufactor/1000
                                ctrl.match_order = match_order
                                ctrl.match_scheme = match_scheme
                                # tasks.append(executor.submit(run_py_metis_con, common, ctrl, logger,task_id))
                                pool.apply_async(run_py_metis_con, (common, ctrl, resultLogFile,))
                                # run_metis_main(common, ctrl)
                                # parts=run_metis_main(common, ctrl)
                                # assignment = {}
                                # for part in parts:
                                #     assignment['LEO'+parts[part][0]]=['LEO{}'.format(n) for n in parts[part]]
                                # analysis(common, assignment, logger)
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print("Sub-process(es) done.")                                
    # for task in as_completed(tasks):
    #     print('{:.3f}: task-{} done'.format(time.time() ,task.result()))                                
    # index=1
    # while index<20:
    #     index+=1
    #     time.sleep(1)
    #     print(time.time())
    #     for task in tasks:
    #         print('{}'.format(task.done()))
        
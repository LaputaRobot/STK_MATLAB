import logging
import os
import shutil
import sys
import time
import multiprocessing
import numpy as np

from PyMetis import run_metis_main, pymetis_parallel
from analysis_result import deploy_in_area, apply_partition, get_avg_flow_setup_time, analysis
from balcon import bal_con_assign, run_balcon_parallel
from config import *
from greedy import greedy_alg1
from metis import gen_metis_file, read_metis_result
from concurrent.futures import ThreadPoolExecutor, as_completed
from util import gen_topology, Common, new_file, get_log_handlers, Ctrl

if __name__ == '__main__':
    files = os.listdir(os.path.join(result_base, 'part/topos'))
    files.sort(key=lambda x: int(x.split('.')[0]))
    t_index = 0
    # executor = ThreadPoolExecutor(max_workers=10)
    # tasks=[]
    start_t = time.time()
    pool=multiprocessing.Pool(processes=8)
    for f in files[:50]:
        t = int(f.split('.')[0])
        t_index += 1
        print('\n\033[1;36m', '+'*77,
              "时隙: {}, {}".format(t_index, t), '+'*77, '\033[0m')
        assignment = None
        if AssignScheme == 'Greedy':
            common = Common(t)
            assignment = greedy_alg1(common)
        elif AssignScheme == 'SamePlane':
            assignment = assignment1
        elif AssignScheme == 'BalCon':
            common = Common(t)
            for mcs in range(1, 10):
                for mssls in range(15, 30):
                    # bal_assign_f = '/home/ygb/result/part/balcon/{}/{}-{}.ass'.format(t, mcs, mssls)
                    bal_log_f = os.path.join(
                        result_base, 'part/balcon/{}/{}-{}.log'.format(t, mcs, mssls))
                    if not os.path.exists(bal_log_f) or Rewrite:
                        pool.apply_async(run_balcon_parallel,
                                         (common, mcs, mssls, bal_log_f,))
                        # run_balcon_parallel(common, mcs, mssls, bal_log_f)
                        # new_file(bal_log_f)
                        # logger = logging.getLogger('{}'.format(bal_log_f))
                        # logger.setLevel(logging.INFO)
                        # handlers = get_log_handlers([LogToFile], bal_log_f)
                        # for handler in handlers:
                        #     logger.addHandler(handler)
                        # initialAssign = assignment2
                        # # f = open(bal_assign_f, 'w')
                        # assignment = bal_con_assign(common, initialAssign, mcs, mssls, logger)
                        # analysis(common, assignment, logger)
                        # f.write(assignment.__str__())
                        # f.close()
        elif AssignScheme == 'METIS':
            metisFile = os.path.join(result_base,'part/MetisTopos/{}'.format(t))
            common = None
            if not os.path.exists(metisFile):
                common = Common(t)
                gen_metis_file(common)
            scheme = 'src'
            ass_set = set()
            for p in range(8, 9):
                for ufactor in range(500, 1700, 100):
                    for contig in ['-contig', '']:
                        # for minconn in ['-minconn', '']:
                        for seed in range(10):
                            part_file = os.path.join(result_base,'part/metis/part/{}/{}-{}-{}{}.part'.format(
                                t, p, ufactor, seed, contig)) 
                            if os.path.exists(part_file) and not Rewrite:
                                continue
                            if common is None:
                                common = Common(t)
                            log_file = os.path.join(result_base,'part/metis/log/{}/{}-{}-{}{}.log'.format(
                                t, p, ufactor, seed, contig))
                            new_file(log_file)
                            new_file(part_file)
                            logger = logging.getLogger(
                                '{}-{}-{}-{}-{}'.format(t, p, ufactor, seed, contig))
                            logger.setLevel(logging.INFO)
                            handlers = get_log_handlers([LogToFile], log_file)
                            for handler in handlers:
                                logger.addHandler(handler)
                            # cmd = 'gpmetis {} {} -ufactor={} {} |tee -a {}'.format(metisFile, p, ufactor, contig,
                            #                                                        resultLogFile)
                            cmd = 'gpmetis {} {} -ufactor={} -seed={} {}  > {}'.format(metisFile, p, ufactor, seed, contig,
                                                                                       log_file)
                            resultFile = '{}.part.{}'.format(metisFile, p)
                            os.system(cmd)
                            shutil.move(resultFile, part_file)
                            assignment = read_metis_result(part_file, ass_set)
                            if assignment is None:
                                continue
                            analysis(common, assignment, logger)
            print('ass set len: {}'.format(len(ass_set)))
        elif AssignScheme == 'PyMetis':
            common = Common(t)
            for p in range(8, 9):
                for ufactor in range(500, 1700, 100):
                    for match_order in [MO_SRC, MO_LoadWeiLoad, MO_SumWei, MO_Wei]:
                    # for match_order in [MO_SRC]:
                        for match_scheme in [MS_SRC, MS_WeiDif, MS_WeiLoad]:
                        # for match_scheme in [MS_SRC]:
                            for contig in ['-contig', '']:
                                for seed in range(10):
                                    log_file = os.path.join(result_base,'part/pymetis/{}/{}-{}-{}-{}-{}{}.log'.format(
                                        t, p, ufactor, match_order, match_scheme, seed, contig))
                                    if os.path.exists(log_file) and not Rewrite:
                                        continue
                                    new_file(log_file)
                                    ctrl = Ctrl()
                                    ctrl.contiguous = True if contig == '-contig' else False
                                    ctrl.nparts = p
                                    ctrl.un_factor = 1 + ufactor/1000
                                    ctrl.match_order = match_order
                                    ctrl.match_scheme = match_scheme
                                    ctrl.rng= np.random.default_rng(seed)
                                    # tasks.append(executor.submit(run_py_metis_con, common, ctrl, logger,task_id))
                                    pool.apply_async(pymetis_parallel, (common, ctrl, log_file,))
                                    # pymetis_parallel(common, ctrl, log_file)
                                    # run_metis_main(common, ctrl)
                                    # parts=run_metis_main(common, ctrl)
                                    # assignment = {}
                                    # for part in parts:
                                    #     assignment['LEO'+parts[part][0]]=['LEO{}'.format(n) for n in parts[part]]
                                    # analysis(common, assignment, logger)
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print("Sub-process(es) done. cost: {}".format(time.time()-start_t))
    #
    #
    # for task in as_completed(tasks):
    #     print('{:.3f}: task-{} done'.format(time.time() ,task.result()))
    # index=1
    # while index<20:
    #     index+=1
    #     time.sleep(1)
    #     print(time.time())
    #     for task in tasks:
    #         print('{}'.format(task.done()))

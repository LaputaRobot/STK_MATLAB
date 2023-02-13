import json
import os
import shutil
import sys
import logging
import time
import multiprocessing
sys.path.append("..") 
from analysis_result import analysis
from metis import read_metis_result
from util import Common,get_log_handlers,new_file ,pformat_with_indent ,pformat,get_time
from config import LogToFile,LogToScreen,RepartScheme,result_base
from balcon import bal_con_assign


def get_pre_part(t):
    pre_part_file= os.path.join(result_base, 'repart/prepart/{}'.format(t))
    if os.path.exists(pre_part_file):
        return pre_part_file
    sum_file= os .path.join(result_base, 'part/metis/src/{}/-{}-sum.json'.format(t,t))
    pre_graph_file = os.path.join(result_base, 'part/MetisTopos/{}'.format(t))
    args= {}
    cmd = ''
    with open(sum_file, 'r') as f:
        result = json.load(f)
        args=result['args']
    contig='-contig' if args['contig'] else ''
    minconn='-minconn' if args['minconn'] else ''
    part_file = '{}.part.{}'.format(pre_graph_file, args['part'])
    
    cmd = 'gpmetis {} {} -ufactor={} {} {} '.format(pre_graph_file, args['part'], args['ufactor'], contig, minconn)
    os.system(cmd)
    shutil.move(part_file, pre_part_file)
    return  pre_part_file

def get_pre_ass(t):
    sum_file=os.path.join(result_base, 'part/metis/src/{}/-{}-sum.json'.format(t,t))
    pre_ass = {}
    with open(sum_file, 'r') as f:
        result = json.load(f)
        for con in result['assign']:
            pre_ass[con]=result['assign'][con]['switches']
    return  pre_ass


def repart(t1,t2,nparts,ubvec,itr,log_level,seed,pre_part_file,with_vsize=False):
    new_graph_file = os.path.join(result_base,'part/MetisTopos/{}'.format(t2))
    new_part_file= os.path.join(result_base,'repart/newpart/{}/{}-{}-{}.part'.format(t2,ubvec,itr,seed))
    new_file(new_part_file)
    os.environ["LD_LIBRARY_PATH"]='/home/ygb/parlocal/lib'
    # os.environ["LD_LIBRARY_PATH"]='/home/ygb'
    # tt2=time.time()
    cmd ='ad_repart {} {} {} 3 0 {} {} {} {} 2 {}'.format(
            new_graph_file, nparts, pre_part_file ,ubvec/10,itr/1000,log_level,seed,new_part_file)
    print(cmd)
    os.system(cmd)
    # print('repart cost: {}'.format(time.time()-tt2))
    return new_part_file


def parmetis_repart(common,pre_t, t,  ubvec, itr, seed, pre_part_file):
    # t1=time.time()
    new_part_file=repart(pre_t, t, 8, ubvec, itr, 0, seed, pre_part_file)
    new_part_file= os.path.join(result_base,'repart/newpart/{}/{}-{}-{}.part'.format(t,ubvec,itr,seed))
    assignment = read_metis_result(new_part_file)
    result_log= os.path.join(result_base,'repart/resultlog/{}/{}-{}-{}.log'.format(t,ubvec,itr,seed))
    new_file(result_log)
    logger = logging.getLogger('{}'.format(result_log))
    logger.setLevel(logging.INFO)
    handlers = get_log_handlers([ LogToFile],result_log)
    for handler in handlers:
        logger.addHandler(handler)
    print('{}-{}-{}.log'.format(ubvec,itr,seed))
    analysis(common, assignment, logger)

def parmetis_analysis():
    new_part_file= os.path.join(result_base,'repart/newpart/{}/{}-{}-{}.part'.format(t,ubvec,itr,seed))
    assignment = read_metis_result(new_part_file, ass_dic)
    result_log= os.path.join(result_base,'repart/resultlog/{}/{}-{}-{}.log'.format(t,ubvec,itr,seed))
    new_file(result_log)
    logger = logging.getLogger('{}'.format(result_log))
    logger.setLevel(logging.INFO)
    handlers = get_log_handlers([ LogToFile],result_log)
    for handler in handlers:
        logger.addHandler(handler)
    analysis(common, assignment, logger)


if __name__ == "__main__":
    files = os.listdir(os.path.join(result_base,'part/topos'))
    files.sort(key=lambda x: int(x.split('.')[0]))
    time_slots = [int(f.split('.')[0]) for f in files]
    t_index = 1
    start_t =time.time()
    # pool=multiprocessing.Pool(processes=8)
    for t in time_slots[1:10]:
        pre_t=time_slots[t_index-1]
        # ass_dic=multiprocessing.Manager().dict()
        t_index += 1
        common = Common(t)
        slot_start = time.time()
        if RepartScheme =='parmetis':
            pre_part_file = get_pre_part(pre_t)
            ubvec_min=16
            ubvec_max=20
            itr_min = 10
            itr_max = 200
            seed_min = 1
            seed_max = 10
            new_graph_file = os.path.join(result_base,'part/MetisTopos/{}'.format(t))
            part_file_dir= os.path.join(result_base,'repart/newpart/{}/'.format(t))
            os.environ["LD_LIBRARY_PATH"]='/home/ygb/parlocal/lib'
            cmd = 'repart {} {} {} 3 0 {} {} {} {} {} {} {} 2 {}'.format(
                new_graph_file, 8, pre_part_file ,ubvec_min,ubvec_max,itr_min,itr_max,0,seed_min,seed_max,part_file_dir)
            print(cmd)
            cmd_t1=time.time()
            os.system(cmd)
            print('cmd cost: {}'.format(time.time()-cmd_t1))
            for ubvec in range(ubvec_min,ubvec_max+1): #1.5 -> 2.2
                for itr in range(itr_min,itr_max+10,10): # 0.01 -0.01-> 0.2
                    for seed in range(seed_min,seed_max+1):
                        # pool.apply_async(repart,(pre_t, t, 8, ubvec, itr, 0, seed, pre_part_file,))
                        # pool.apply_async(parmetis_repart, (common, pre_t, t, ubvec, itr, seed, pre_part_file,))
                        # parmetis_repart(common, pre_t, t, ubvec, itr, seed, pre_part_file)
                        # start_t = time.time()
                        # print('{:>2d} {:>5d} {:>2d}'.format(ubvec,itr,seed))
                        # TODO 使用多进程进行分析
                        new_part_file= os.path.join(result_base,'repart/newpart/{}/{}-{}-{}.part'.format(t,ubvec,itr,seed))
                        assignment = read_metis_result(new_part_file)
                        result_log= os.path.join(result_base,'repart/resultlog/{}/{}-{}-{}.log'.format(t,ubvec,itr,seed))
                        new_file(result_log)
                        logger = logging.getLogger('{}'.format(result_log))
                        logger.setLevel(logging.INFO)
                        handlers = get_log_handlers([ LogToFile],result_log)
                        for handler in handlers:
                            logger.addHandler(handler)
                        if assignment is None:
                            if handlers.__len__()>1:
                                logger.removeHandler(handlers[1])
                            logger.info('avg_setup_time: {:>6.2f}\n'.format(300))
                            continue
                        print('{}-{}-{}.log'.format(ubvec,itr,seed))
                        analysis(common, assignment, logger)
        print('slot cost: {}'.format(time.time()-slot_start))
        if RepartScheme == 'balcon':
            pre_ass = get_pre_ass(pre_t)
            for MCS in range(1,10):
                for MSSLS in range(15,30):
                    start_t=time.time()
                    result_log=os.path.join(result_base,'repart/balcon/{}/{}-{}.log'.format(t,MCS,MSSLS))
                    new_file(result_log)
                    logger = logging.getLogger('{}'.format(result_log))
                    logger.setLevel(logging.INFO)
                    handlers = get_log_handlers([LogToScreen,LogToFile],result_log)
                    for handler in handlers:
                        logger.addHandler(handler)
                    # for part in pre_ass.values():
                    #     print(part)
                    # print()
                    assignment = bal_con_assign(common, pre_ass, MCS, MSSLS, logger)
                    # for part in assignment.values():
                    #     print(part)
                    # print()
                    analysis(common, assignment, logger)
                    # cost_t = time.time()-start_t
                    # print('cost time: {}'.format(cost_t))
    # pool.close()
    # pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print("Sub-process(es) done in {}.".format(time.time()-start_t)) 
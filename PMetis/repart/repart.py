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
    new_file(pre_part_file)
    sum_file= os .path.join(result_base, 'part/metis/log/{}/-{}-sum.json'.format(t,t))
    # pre_graph_file = os.path.join(result_base, 'part/MetisTopos/{}'.format(t))
    args= {}
    cmd = ''
    with open(sum_file, 'r') as f:
        result = json.load(f)
        args = result['args']
    contig='-contig' if args['contig'] else ''
    # minconn='-minconn' if args['minconn'] else ''
    part_file =os.path.join(result_base, 'part/metis/part/{}/{}-{}-{}{}.part'.format(t, args['part'],args['ufactor'],args['seed'],contig)) 
    
    # cmd = 'gpmetis {} {} -ufactor={} {} {} '.format(pre_graph_file, args['part'], args['ufactor'], contig, minconn)
    # os.system(cmd)
    shutil.copy(part_file, pre_part_file)
    return  pre_part_file

def get_pre_ass(t):
    sum_file=os.path.join(result_base, 'part/metis/log/{}/-{}-sum.json'.format(t,t))
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

def parmetis_analysis(common,t,ubvec,itr,seed, ass_set):
    new_part_file= os.path.join(result_base,'repart/newpart/{}/{}-{}-{}.part'.format(t,ubvec,itr,seed))
    assignment = read_metis_result(new_part_file, ass_set)
    if assignment is None:
        return
    result_log= os.path.join(result_base,'repart/resultlog/{}/{}-{}-{}.log'.format(t,ubvec,itr,seed))
    new_file(result_log)
    logger = logging.getLogger('{}'.format(result_log))
    logger.setLevel(logging.INFO)
    handlers = get_log_handlers([ LogToFile],result_log)
    for handler in handlers:
        logger.addHandler(handler)
    analysis(common, assignment, logger)

def balcon_repart(common, pre_ass, MCS, MSSLS, t):
    result_log=os.path.join(result_base,'repart/balcon/{}/{}-{}.log'.format(t,MCS,MSSLS))
    new_file(result_log)
    logger = logging.getLogger('{}'.format(result_log))
    logger.setLevel(logging.INFO)
    handlers = get_log_handlers([LogToFile],result_log)
    for handler in handlers:
        logger.addHandler(handler)
    assignment = bal_con_assign(common, pre_ass, MCS, MSSLS, logger)
    analysis(common, assignment, logger)

if __name__ == "__main__":
    files = os.listdir(os.path.join(result_base,'part/topos'))
    files.sort(key=lambda x: int(x.split('.')[0]))
    time_slots = [int(f.split('.')[0]) for f in files]
    t_index = 1
    start_t = time.time()
    pool=multiprocessing.Pool(processes=8)
    for t in time_slots[10:50]:
        pre_t=time_slots[t_index-1]
        # ass_dic=multiprocessing.Manager().dict()
        t_index += 1
        slot_start = time.time()
        common = Common(t)
        if RepartScheme =='parmetis':
            pre_part_file = get_pre_part(pre_t)
            ubvec_min=16
            ubvec_max=22
            itr_min = 10
            itr_max = 300
            seed_min = 1
            seed_max = 10
            new_graph_file = os.path.join(result_base,'part/MetisTopos/{}'.format(t))
            part_file_dir= os.path.join(result_base,'repart/newpart/{}/'.format(t))
            os.makedirs(part_file_dir, exist_ok=True)
            os.environ["LD_LIBRARY_PATH"]='/home/ygb/parlocal/lib'
            cmd = 'repart {} {} {} 3 0 {} {} {} {} {} {} {} 2 {}'.format(
                new_graph_file, 8, pre_part_file ,ubvec_min,ubvec_max,itr_min,itr_max,0,seed_min,seed_max,part_file_dir)
            print(cmd)
            cmd_t1=time.time()
            os.system(cmd)
            print('cmd cost: {}'.format(time.time()-cmd_t1))
            ass_set=set()
            for ubvec in range(ubvec_min,ubvec_max+1): #1.5 -> 2.2
                for itr in range(itr_min,itr_max+10,10): # 0.01 -0.01-> 0.2
                    for seed in range(seed_min,seed_max+1):
                        # pool.apply_async(parmetis_analysis, (common, t, ubvec, itr, seed,))
                        parmetis_analysis(common, t, ubvec, itr, seed,ass_set)
                        # pool.apply_async(repart,(pre_t, t, 8, ubvec, itr, 0, seed, pre_part_file,))
                        # pool.apply_async(parmetis_repart, (common, pre_t, t, ubvec, itr, seed, pre_part_file,))
                        # parmetis_repart(common, pre_t, t, ubvec, itr, seed, pre_part_file)
                        # start_t = time.time()
                        # print('{:>2d} {:>5d} {:>2d}'.format(ubvec,itr,seed))
            print('set len: {}'.format(len(ass_set)))          
        # print('slot cost: {}'.format(time.time()-slot_start))
        print('slot---------------{}----{}------------'.format(pre_t,t))
        if RepartScheme == 'balcon':
            pre_ass = get_pre_ass(pre_t)
            for MCS in range(1,10):
                for MSSLS in range(15,30):
                    # assignment = bal_con_assign(common, pre_ass, MCS, MSSLS, logger)
                    # analysis(common, assignment, logger)

                    pool.apply_async(balcon_repart,(common, pre_ass, MCS, MSSLS, t,))
                    # balcon_repart(common, pre_ass, MCS, MSSLS, t)
                    # cost_t = time.time()-start_t
                    # print('cost time: {}'.format(cost_t))
    pool.close()
    pool.join()   #??????join??????????????????close????????????????????????????????????close?????????????????????????????????pool,join?????????????????????????????????
    print("Sub-process(es) done in {}.".format(time.time()-start_t)) 
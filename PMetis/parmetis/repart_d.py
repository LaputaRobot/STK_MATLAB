import json
import os
import shutil
import sys
import logging
sys.path.append("..") 
from analysis_result import analysis
from metis import read_metis_result
from util import Common,get_log_handlers,new_file
from config import LogToFile,LogToScreen,RepartScheme
from balcon import bal_con_assign


def get_pre_part(t):
    pre_part_file='/home/ygb/STK_MATLAB/PMetis/parmetis/prepart/{}'.format(t)
    if os.path.exists(pre_part_file):
        return pre_part_file
    sum_file='/home/ygb/STK_MATLAB/PMetis/metis/src/{}/-{}-sum.json'.format(t,t)
    pre_graph_file = '/home/ygb/STK_MATLAB/PMetis/MetisTopos/{}'.format(t)
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
    sum_file='/home/ygb/STK_MATLAB/PMetis/metis/src/{}/-{}-sum.json'.format(t,t)
    pre_ass= {}
    with open(sum_file, 'r') as f:
        result = json.load(f)
        for con in result['assign']:
            pre_ass[con]=result['assign'][con]['switches']
    return  pre_ass

def repart(t1,t2,nparts,ubvec,itr,log_level,seed):
    pre_part_file = get_pre_part(t1)
    new_graph_file = '/home/ygb/STK_MATLAB/PMetis/MetisTopos/{}'.format(t2)
    new_part_file='/home/ygb/STK_MATLAB/PMetis/parmetis/newpart/{}/{}-{}-{}.part'.format(t2,ubvec,itr,seed)
    new_file(new_part_file)
    os.environ["LD_LIBRARY_PATH"]='/home/ygb/parlocal/lib'
    # os.environ["LD_LIBRARY_PATH"]='/home/ygb'
    cmd ='repart {} {} {} 3 0 {} {} {} {} 2 {} > /dev/null'.format(
        new_graph_file, nparts, pre_part_file ,ubvec/10,itr/1000,log_level,seed,new_part_file
    )
    os.system(cmd)
    return new_part_file


if __name__ == "__main__":
    dirname, filename = os.path.split(os.path.abspath(__file__))
    files = os.listdir(os.path.join(dirname,'../topos'))
    files.sort(key=lambda x: int(x.split('.')[0]))
    time_slots = [int(f.split('.')[0]) for f in files]
    t_index = 1
    for t in time_slots[1:2]:
        pre_t=time_slots[t_index-1]
        t_index += 1
        common = Common(t)
        if RepartScheme =='parmetis':
            for ubvec in range(15,22): #1.5 -> 2.2
                for itr in range(10,200,10): # 0.01 -0.01-> 0.2
                    for seed in range(1,10):
                        new_part_file=repart(pre_t, t, 8, ubvec, itr, 2047, seed)
                        result_log='resultlog/{}/{}-{}-{}.log'.format(t,ubvec,itr,seed)
                        new_file(result_log)
                        assignment = read_metis_result(file_name=new_part_file)
                        logger = logging.getLogger('{}'.format(result_log))
                        logger.setLevel(logging.INFO)
                        handlers = get_log_handlers([LogToScreen,LogToFile],result_log)
                        for handler in handlers:
                            logger.addHandler(handler)
                        analysis(common, assignment, logger)
        if RepartScheme == 'balcon':
            pre_ass = get_pre_ass(pre_t)
            for MCS in range(1,10):
                for MSSLS in range(15,30):
                    result_log='balcon/{}/{}-{}.log'.format(t,MCS,MSSLS)
                    new_file(result_log)
                    logger = logging.getLogger('{}'.format(result_log))
                    logger.setLevel(logging.INFO)
                    handlers = get_log_handlers([LogToScreen,LogToFile],result_log)
                    for handler in handlers:
                        logger.addHandler(handler)
                    assignment = bal_con_assign(common, pre_ass, MCS, MSSLS, logger)
                    analysis(common, assignment, logger)
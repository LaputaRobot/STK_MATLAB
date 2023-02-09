import json
import os
import shutil
import sys
import logging
sys.path.append("..") 
from analysis_result import analysis
from metis import read_metis_result
from util import Common,get_log_handlers
from config import LogToFile,LogToScreen

def get_pre_part(t):
    pre_part_file='/home/ygb/STK_MATLAB/PMetis/parmetis/prepart/{}'.format(t)
    sum_file='/home/ygb/STK_MATLAB/PMetis/metis/src/{}/-{}-sum.json'.format(t,t)
    pre_graph_file = '/home/ygb/STK_MATLAB/PMetis/MetisTopos/{}'.format(t)
    args= {}
    cmd = ''
    with open(sum_file, 'r') as f:
        result = json.load(f)
        args=result['args']
        contig='-contig' if args['contig'] else ''
        minconn='-minconn' if args['minconn'] else ''
        cmd = 'gpmetis {} {} -ufactor={} {} {} '.format(pre_graph_file, args['part'], args['ufactor'], contig, minconn)
    part_file = '{}.part.{}'.format(pre_graph_file, args['part'])
    os.system(cmd)
    shutil.move(part_file, pre_part_file)
    return  pre_part_file
    # assignment = read_metis_result(
    #     file_name=assignmentFile)
    # analysis(common, assignment, logger)

def repart(t1,t2,nparts,ubvec,itr,log_level,seed):
    pre_part_file = get_pre_part(t1)
    new_graph_file = '/home/ygb/STK_MATLAB/PMetis/MetisTopos/{}'.format(t2)
    new_part_file='/home/ygb/STK_MATLAB/PMetis/parmetis/newpart/{}'.format(t2)
    cmd='repart {} {} {} 3 0 {} {} {} {} 2 {}'.format(
        new_graph_file, nparts, pre_part_file ,ubvec,itr,log_level,seed,new_part_file
    )
    os.system(cmd)
    return new_part_file


if __name__ == "__main__":
    t1=0
    t2=10
    new_part_file=repart(t1,t2,8,2.0,1.0,2047,1)
    assignment = read_metis_result(file_name=new_part_file)
    logger = logging.getLogger('l1')
    logger = logging.getLogger('{}'.format(t2))
    logger.setLevel(logging.INFO)
    handlers = get_log_handlers([LogToScreen],None)
    for handler in handlers:
        logger.addHandler(handler)
    common = Common(t2)
    analysis(common, assignment, logger)
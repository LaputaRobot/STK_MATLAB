import json
import os
import shutil
import sys
import logging
sys.path.append("..") 
from analysis_result import analysis
from metis import read_metis_result
from util import Common,get_log_handlers,new_file
from config import LogToFile,LogToScreen



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
        for MCS in range(3,5):
            for MSSLS in range(24,26):
                result_log='balcon/{}/{}-{}.log'.format(t,MCS,MSSLS)
                new_file(result_log)
                logger = logging.getLogger('{}'.format(result_log))
                logger.setLevel(logging.INFO)
                handlers = get_log_handlers([LogToScreen,LogToFile],result_log)
                for handler in handlers:
                    logger.addHandler(handler)
                pre_ass = get_pre_ass(pre_t)
                assignment = bal_con_assign(common, pre_ass, logger)
                analysis(common, assignment, logger)

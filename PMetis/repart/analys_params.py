import os
import sys

from pprint import pprint
sys.path.append("..") 
from config import result_base

def analysis(dir_path):
    path=os.path.join(result_base,dir_path)
    times=os.listdir(path)
    times=sorted(times, key=lambda x: int(x))
    best_times = {'ubvec':{},'itr':{},'seed':{}}
    for t in times[1:]:
        print('time_slot: {}'.format(t)) 
        result = {}
        t_path = os.path.join(path,t)
        files=os.listdir(t_path)
        files=sorted(files)
        for f in files:
            if f.endswith('log'):
                args=f.split('.')[0]
                cmd = "cd {} && tail -n 1 {} |awk '{{print $NF}}'".format(t_path,f)
                # print(cmd)
                delay = os.popen(cmd).read()
                if delay in result:
                    result[delay].append(args)
                else:
                    result[delay] =[args]
        print('diff delay: {}'.format(len(result)))
        new_result = {}
        sorted_keys=sorted(list(result.keys()))
        for key in sorted_keys[:5]:
            new_result[key]=result[key]
        for args_str in result[sorted_keys[0]]:
            args=args_str.split('-')
            ubvec = args[0]
            itr = args[1]
            seed = args[2]
            if ubvec in  best_times['ubvec']:
                best_times['ubvec'][ubvec]+=1
            else:
                best_times['ubvec'][ubvec]=1
            if itr in  best_times['itr']:
                best_times['itr'][itr]+=1
            else:
                best_times['itr'][itr]=1
            if seed in  best_times['seed']:
                best_times['seed'][seed]+=1
            else:
                best_times['seed'][seed]=1
        pprint(new_result, width=300, compact= True)
        print('\n'*3)
    new_dict={}
    for arg in best_times:
        new_dict[arg]= dict(sorted(best_times[arg].items(), key=lambda item: item[1],reverse=True))
    print(new_dict)
        


if __name__ == "__main__":
    analysis('repart/resultlog')

    """{'ubvec': {'17': 1860, '18': 1523, '19': 1374, '16': 1222, '20': 901, '21': 873, '15': 290}, 
    'itr': {'80': 431, '100': 404, '110': 404, '120': 404, '130': 404, '140': 404, '150': 404, '160': 404, 
    '90': 404, '170': 402, '180': 402, '190': 402, '50': 400, '70': 395, '60': 393, '40': 389, '20': 373,
     '30': 372, '10': 265, '10000': 58, '11000': 58, '12000': 58, '13000': 58, '14000': 58, '15000': 58,
      '16000': 58, '17000': 58, '18000': 58, '19000': 58, '1000': 1, '400': 1, '500': 1, '600': 1, '700': 1, '800': 1, '900': 1}, 
      'seed': {'1': 953, '4': 928, '7': 904, '3': 896, '8': 895, '5': 875, '2': 874, '6': 863, '9': 855}}
    """
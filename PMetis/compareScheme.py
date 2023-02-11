import math
import os
import json

from config import Rewrite, result_base


def get_base_dir(scheme):
    scheme_dir = ''
    if scheme == 'metis':
        scheme_dir = 'part/metis/src'
    if scheme == 'balcon':
        scheme_dir = 'part/balcon'
    if scheme == 'pymetis':
        scheme_dir = 'part/pymetis'
    if scheme == 'parmetis':
        scheme_dir =  'repart/resultlog'
    if scheme == 'balcon-re':
        scheme_dir = 'repart/balcon'
    return os.path.join(result_base, scheme_dir)


def get_sum_result(scheme):
    base_dir = get_base_dir(scheme)
    times = os.listdir(base_dir)
    result = {}
    for t in times:
        result[t] = {"sum_con_loads": 0, "max_con_load": 0, "avg_setup_time": math.inf,
                     "assign": {}, "args": {}}
        result_files = os.listdir(os.path.join(base_dir, t))
        if '-{}-sum.json'.format(t) in result_files and not Rewrite:
            continue
        for file in result_files:
            if str.endswith(file, 'log'):
                args_list = str.split(file[:-4], '-')
                args = {}
                if scheme == 'metis':
                    args = {"part": args_list[0], "ufactor": args_list[1], "contig": 'contig' in file,
                    'minconn': 'minconn' in file}
                    if args['minconn']:
                        continue
                if scheme == 'balcon' or scheme == 'balcon-re':
                    args = {"MCS": args_list[0], "MSSLS": args_list[1]}
                if scheme == 'pymetis':
                    args = {"part": args_list[0], "ufactor": args_list[1],'match_order': args_list[2],'match_scheme':args_list[3], "contig": 'contig' in file}
                if scheme == 'parmetis':
                    args = {"ubvec":args_list[0],"itr":args_list[1],"seed":args_list[2]}
                sum_load = 0
                max_load = 0
                delay = 0
                f = open(os.path.join(base_dir, t, file), 'r')
                lines = f.readlines()
                ass = {}
                for line in lines:
                    if 'LEO' in line and '[' in line:
                        cols = line.split(':')
                        con = cols[0]
                        load = float(cols[1])
                        switches = eval(cols[2])
                        ass[con] = {"load": load, "switches": switches}
                    if 'sum_con_loads' in line:
                        sum_load = float(line.split(':')[1])
                    if 'max_con_loads' in line:
                        max_load = float(line.split(':')[1])
                    if 'avg_setup_time' in line:
                        delay = float(line.split(' ')[-1])
                if delay < result[t]['avg_setup_time'] and max_load < 100:
                    result[t] = {"sum_con_loads": sum_load, "max_con_load": max_load,
                                 "avg_setup_time": delay, "assign": ass, "args": args}
        sum_file = open(os.path.join(base_dir, t, '-{}-sum.json'.format(t)), 'w')
        sum_file.write(json.dumps(result[t], indent=4, separators=(',', ': ')))
        sum_file.close()


def compare(schemes):
    files = os.listdir(os.path.join(result_base, 'part/MetisTopos') )
    files.sort(key=lambda x: int(x.split('.')[0]))
    print('{:>12} {:>12}, {:>12}, {:>12}'.format(
        '', 'sum_load', 'max_load', 'delay'))
    avg_result = {}
    sum_result = {}
    for scheme in schemes:
        sum_result[scheme] = 0
    t_index = 0 
    for t in files[1:50]:
        pre_t = files[t_index]
        t_index += 1
        best_scheme = ''
        setup_time = math.inf
        for scheme in schemes:
            sum_file = os.path.join(get_base_dir(
                scheme), t, '-{}-sum.json'.format(t))
            with open(sum_file, 'r') as f:
                result = json.load(f)
                print('{:>12}: {:>12.2f}, {:>12.2f}, {:>12.2f}, {}'.format(scheme, result['sum_con_loads'],
                                                                           result['max_con_load'],
                                                                           result['avg_setup_time'],
                                                                           result['args']))
                sum_result[scheme]+= result['avg_setup_time']                                             
                if result['avg_setup_time'] < setup_time:
                    best_scheme = scheme
                    setup_time = result['avg_setup_time']
        print(
            '------- slot {:>7}, best scheme is ----- {}\n'.format(t, best_scheme))
    for scheme in schemes:
        avg_result[scheme] = sum_result[scheme]/50
    print(avg_result)

def get_node_loads(t):
    graph_file = os.path.join(result_base, 'part/MetisTopos/{}'.format(t)) 
    with open(graph_file,'r') as f:
        lines = f.read().splitlines()
    node_loads= {}
    node = 0
    for line in lines[1:]:
        node_loads[node] = int(line)
    return node_loads

def get_repart_cost(t,pre_t,scheme,args, node_loads):
    """
    获取每种方案改变的节点
    获取节点的负载
    """
    
    # TODO 分析两种方案的迁移成本 
    cost = 0
    if scheme == 'parmetis':
        pre_part_file = os.path.join(result_base, 'repart/prepart/{}'.format(pre_t)) 
        new_part_file =  os.path.join(result_base, 'repart/newpart/{}/{}-{}-{}.part'.format( t,args['ubvec'], args['itr'], args['seed']))
        with open(pre_part_file,'r') as f:
            lines_pre = f.read().splitlines()
        with open(new_part_file, 'r') as f:
            lines_new = f.read().splitlines()
        assert len(line_new)==72
        node = 0
        for index in range(len(lines_pre)):
            if lines_pre[index]!= lines_new[index]:
                cost += node_loads[index]
    if scheme == 'balcon-re':
        # TODO 修改balcon返回被迁移的节点，并计算其迁移成本
        pass

    

def repart_compare(schmes):
    files = os.listdir(os.path.join(result_base, 'repart/newpart'))
    files.sort(key=lambda x: int(x.split('.')[0]))
    print('{:>12} {:>12}, {:>12}, {:>12}'.format(
        '', 'sum_load', 'max_load', 'delay'))
    avg_result = {}
    sum_result = {}
    for scheme in schemes:
        sum_result[scheme] = 0
    for t in files:
        best_scheme = ''
        setup_time = math.inf
        for scheme in schemes:
            sum_file = os.path.join(get_base_dir(
                scheme), t, '-{}-sum.json'.format(t))
            with open(sum_file, 'r') as f:
                result = json.load(f)
                print('{:>12}: {:>12.2f}, {:>12.2f}, {:>12.2f}, {}'.format(scheme, result['sum_con_loads'],
                                                                           result['max_con_load'],
                                                                           result['avg_setup_time'],
                                                                           result['args']))
                sum_result[scheme]+= result['avg_setup_time']                                             
                if result['avg_setup_time'] < setup_time:
                    best_scheme = scheme
                    setup_time = result['avg_setup_time']
        print(
            '------- slot {:>7}, best scheme is ----- {}\n'.format(t, best_scheme))
    for scheme in schemes:
        avg_result[scheme] = sum_result[scheme]/50
    print(avg_result)
    # TODO 总负载*时延差-节点迁移中断时间*被迁移的负载 两值比较

if __name__ == '__main__':
    # schemes = ['metis', 'balcon','pymetis']
    schemes = ['parmetis','balcon-re']
    for scheme in schemes:
        get_sum_result(scheme)
    repart_compare(schemes)

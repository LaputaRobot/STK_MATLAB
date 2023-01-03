import math
import os

from PMetis.config import Rewrite


def get_base_dir(scheme):
    if scheme == 'metis':
        return 'metis/src'
    if scheme == 'balcon':
        return 'balcon'


def get_sum_result(scheme):
    base_dir = get_base_dir(scheme)
    times = os.listdir(base_dir)
    result = {}
    for t in times:
        result[t] = {'sum_con_loads': 0, 'max_con_load': 0, 'avg_setup_time': math.inf,
                     'assign': {'con': None, 'load': 0, 'switches': []},
                     'args': {'part': 0, 'ufactor': 0, 'contig': True}}
        result_files = os.listdir(os.path.join(base_dir, t))
        if '{}.sum'.format(t) in result_files and not Rewrite:
            continue
        for file in result_files:
            if str.endswith(file, 'log'):
                args_list = str.split(file[:-4], '-')
                args = {}
                if scheme == 'metis':
                    args = {'part': args_list[0], 'ufactor': args_list[1], 'contig': len(args_list) > 2}
                if scheme == 'balcon':
                    args = {'MCS': args_list[0], 'MSSLS': args_list[1]}
                con = ''
                load = 0
                switches = []
                sum_load = 0
                max_load = 0
                delay = 0
                f = open(os.path.join(base_dir, t, file), 'r')
                lines = f.readlines()
                for line in lines:
                    if 'LEO' in line and '[' in line:
                        cols = line.split(':')
                        con = cols[0]
                        load = float(cols[1])
                        switches = eval(cols[2])
                    if 'sum_con_loads' in line:
                        sum_load = float(line.split(':')[1])
                    if 'max_con_loads' in line:
                        max_load = float(line.split(':')[1])
                    if 'avg_setup_time' in line:
                        delay = float(line.split(' ')[-1])
                if delay < result[t]['avg_setup_time'] and max_load<100:
                    result[t] = {'sum_con_loads': sum_load, 'max_con_load': max_load,
                                 'avg_setup_time': delay,
                                 'assign': {'con': con, 'load': load, 'switches': switches},
                                 'args': args}
        sum_file = open(os.path.join(base_dir, t, '{}.sum'.format(t)), 'w')
        sum_file.write(result[t].__str__())
        sum_file.close()


def compare(schemes):
    files = os.listdir('MetisTopos')
    files.sort(key=lambda x: int(x.split('.')[0]))
    print('{:>12} {:>12}, {:>12}, {:>12}'.format('', 'sum_load', 'max_load', 'delay'))
    for t in files[:50]:
        best_scheme = ''
        setup_time = math.inf
        for scheme in schemes:
            sum_file = os.path.join(get_base_dir(scheme), t, '{}.sum'.format(t))
            with open(sum_file, 'r') as f:
                result = eval(f.readlines()[0])
                print('{:>12}: {:>12.2f}, {:>12.2f}, {:>12.2f}'.format(scheme, result['sum_con_loads'],
                                                                       result['max_con_load'],
                                                                       result['avg_setup_time']))
                if result['avg_setup_time'] < setup_time :
                    best_scheme = scheme
                    setup_time = result['avg_setup_time']
        print('------- slot {:>7}, best scheme is ----- {}'.format(t, best_scheme))


if __name__ == '__main__':
    schemes = ['metis', 'balcon']
    for scheme in schemes:
        get_sum_result(scheme)
    compare(schemes)

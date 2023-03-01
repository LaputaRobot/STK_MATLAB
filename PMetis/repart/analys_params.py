from config import result_base
import os
import sys

from pprint import pprint
sys.path.append("..")


def analysis(dir_path):
    path = os.path.join(result_base, dir_path)
    times = os.listdir(path)
    times = sorted(times, key=lambda x: int(x))
    best_times = {'ubvec': {}, 'itr': {}, 'seed': {}}
    for t in times[1:]:
        print('time_slot: {}'.format(t))
        result = {}
        t_path = os.path.join(path, t)
        files = os.listdir(t_path)
        files = sorted(files)
        for f in files:
            if f.endswith('log'):
                args = f.split('.')[0]
                cmd = "cd {} && tail -n 1 {} |awk '{{print $NF}}'".format(
                    t_path, f)
                # print(cmd)
                delay = os.popen(cmd).read()
                if delay in result:
                    result[delay].append(args)
                else:
                    result[delay] = [args]
        print('diff delay: {}'.format(len(result)))
        new_result = {}
        sorted_keys = sorted(list(result.keys()))
        for key in sorted_keys[:5]:
            new_result[key] = result[key]
        for args_str in result[sorted_keys[0]]:
            args = args_str.split('-')
            ubvec = args[0]
            itr = args[1]
            seed = args[2]
            if ubvec in best_times['ubvec']:
                best_times['ubvec'][ubvec] += 1
            else:
                best_times['ubvec'][ubvec] = 1
            if itr in best_times['itr']:
                best_times['itr'][itr] += 1
            else:
                best_times['itr'][itr] = 1
            if seed in best_times['seed']:
                best_times['seed'][seed] += 1
            else:
                best_times['seed'][seed] = 1
        pprint(new_result, width=300, compact=True)
        print('\n'*3)
    new_dict = {}
    for arg in best_times:
        new_dict[arg] = dict(
            sorted(best_times[arg].items(), key=lambda item: item[1], reverse=True))
    print(new_dict)


if __name__ == "__main__":
    analysis('repart/resultlog')

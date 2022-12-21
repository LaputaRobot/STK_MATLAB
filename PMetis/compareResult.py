import argparse
import os


def get_min_delay(file):
    f = open(file, 'r')
    lines = f.readlines()
    arg = ''
    assignment = []
    min_delay = 10000
    min_args = ''
    min_assignment = None
    sum_load = 0
    min_sum_load = 0
    max_load = 0
    min_max_load = 0
    for line in lines:
        if 'ufactor' in line:
            arg = line
        if 'LEO' in line:
            assignment.append(line)
        if 'sum_con_loads' in line:
            sum_load = float(line.split(':')[1])
        if 'max_con_loads' in line:
            max_load = float(line.split(':')[1])
        if 'avg_setup_time' in line:
            delay = float(line.split(' ')[-1])
            if delay < min_delay and max_load < 100 and len(assignment) == 8:
            # if delay < min_delay and max_load < 100:
                min_delay = delay
                min_args = arg
                min_assignment = assignment
                min_max_load = max_load
                min_sum_load = sum_load
            assignment = []
    return min_args, min_assignment, min_sum_load, min_max_load, min_delay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compare two scheme')
    parser.add_argument('--src', type=str, default='src', help='src scheme')
    parser.add_argument('--new', type=str, help='new scheme')
    args = parser.parse_args()
    src_scheme = args.src
    new_scheme = args.new
    files = os.listdir('metisResult1/{}'.format(src_scheme))
    files.sort(key=lambda x: int(x))
    newSumSum = 0
    newSumMax = 0
    newSumDelay = 0
    srcSumSum = 0
    srcSumMax = 0
    srcSumDelay = 0
    src_score = 0
    new_score = 0
    weird_num1 = 0
    weird_num2 = 0
    sun_reason = 0
    for f in files:
        src_args, _, src_sum_load, src_max_load, src_delay = get_min_delay('metisResult1/{}/{}'.format(src_scheme, f))
        new_args, _, new_sum_load, new_max_load, new_delay = get_min_delay('metisResult1/{}/{}'.format(new_scheme, f))
        newSumSum += new_sum_load
        newSumMax += new_max_load
        newSumDelay += new_delay
        srcSumSum += src_sum_load
        srcSumMax += src_max_load
        srcSumDelay += src_delay
        if src_delay > new_delay:
            new_score += 1
        else:
            src_score += 1
        # print( '{:>5}:\n {:>3}, {:>6.2f}, {} {:>3}, {:>6.2f}, {} {}'.format(f, src_scheme, src_delay, src_args,
        # new_scheme, new_delay, new_args, src_delay > new_delay))
        print('--' * 55)
        isWeird = 0
        if (src_sum_load - new_sum_load) * (src_delay - new_delay) <= 0 and (src_max_load - new_max_load) * (
                src_delay - new_delay) <= 0:
            if src_delay - new_delay < 0:
                isWeird = 1
                weird_num1 += (src_delay - new_delay)
            else:
                isWeird = 2
                weird_num2 += (src_delay - new_delay)
        print('{:<70} {:>10} {:>10}'.format(f, isWeird,src_delay - new_delay<0))
        print('--' * 55)
        print('{:>10}  {:>14}, {:>14} | {:>14}, {:>14} |'.format('args:', src_args.split(',')[1],
                                                                 src_args.split(',')[-2],
                                                                 new_args.split(',')[1], new_args.split(',')[-2]))
        print('{:>10}  {:>30} | {:>30} | {:>30.3}'.format('sum_load:', src_sum_load, new_sum_load,
                                                          src_sum_load - new_sum_load))
        print('{:>10}  {:>30} | {:>30} | {:>30.3}'.format('max_load:', src_max_load, new_max_load,
                                                          src_max_load - new_max_load))
        print('{:>10}  {:>30} | {:>30} | {:>30.3}'.format('delay:', src_delay, new_delay, src_delay - new_delay))

    # print('{}Sum: {}, {}Sum {}'.format(src_scheme, srcSum, new_scheme, newSum))
    print()
    print('{:>10}  {:>30.4} | {:>30.4} | {:>30.4}'.format('AvgSum:', srcSumSum / (src_score + new_score),
                                                          newSumSum / (src_score + new_score),
                                                          srcSumSum / (src_score + new_score) - newSumSum / (
                                                                  src_score + new_score)))
    print('{:>10}  {:>30.4} | {:>30.4} | {:>30.4}'.format('AvgMax:', srcSumMax / (src_score + new_score),
                                                          newSumMax / (src_score + new_score),
                                                          srcSumMax / (src_score + new_score) - newSumMax / (
                                                                  src_score + new_score)))
    print('{:>10}  {:>30.4} | {:>30.4} | {:>30.4}'.format('AvgDelay:', srcSumDelay / (src_score + new_score),
                                                          newSumDelay / (src_score + new_score),
                                                          srcSumDelay / (src_score + new_score) - newSumDelay / (
                                                                  src_score + new_score)))
    print('rate: {:>5.3f}'.format(new_score / (src_score + new_score)))
    print('weird num: {:>5.3f} {:>5.3f} '.format(weird_num1, weird_num2))

# from gurobipy import *
import sys

import gurobipy as gp
from itertools import product

from gurobipy import GRB

from Python.genSatNet import get_all_distance, gen_init_topology, get_pairs_load, get_pairs_delay

num_sw = 72
num_con = 8
products = list(product(range(num_sw), range(num_sw)))
tri_products = list(product(products, range(num_sw)))


def getIndex(string):
    i = int(string[3])
    j = int(string[4])
    return (i - 1) * 10 + j - i


Distance_Dict = get_all_distance()
G = gen_init_topology(Distance_Dict)
pairs_load = get_pairs_load(G)
start = 0
this = 0
pairs_load_sing = {}
for pair in pairs_load:
    if this >= start and pairs_load[pair] != 0:
        pairs_load_sing[pair] = pairs_load[pair]
    this += 1
    if this > 71:
        this = 0
        start += 1

# print(pairs_load_sing)
# sys.exit(0)
pairs_path_delay = get_pairs_delay(G)
pairs_path_delay_sing = {}
start = 0
this = 0
for pair in pairs_path_delay:
    if this >= start:
        pairs_path_delay_sing[pair] = pairs_path_delay[pair]
    this += 1
    if this > 71:
        this = 0
        start += 1
# print(pairs_path_delay_sing)

# # print(len(pairs_path_delay_sing))
pairs_delay = {}
for pair, p_d in pairs_path_delay.items():
    pairs_delay[(getIndex(pair[0]), getIndex(pair[1]))] = p_d[1]
f = open('paths.txt', 'w')
for pair in pairs_load_sing:
    if pair == ('LEO12', 'LEO14'):
        print(pairs_path_delay[('LEO12', 'LEO13')])
    f.write(pair.__str__() + "  " + pairs_path_delay[pair][1].__str__() + ":  ")
    for node in pairs_path_delay_sing[pair][0]:
        f.write("" + getIndex(node).__str__() + '->')
    f.write('\n')
f.close()
sys.exit(0)
# sys.exit(0)


m = gp.Model('globe')

x_sw_con = m.addVars(products, vtype=GRB.BINARY, name='assign')
y_con = m.addVars(num_sw, vtype=GRB.BINARY, name='deployment')
z_c_s_k = m.addVars(tri_products, vtype=GRB.BINARY, name='isSame')

m.addConstr(gp.quicksum(y_con[c] for c in range(num_sw)) == num_con, name='constr4')
m.addConstrs((x_sw_con[(sw, con)] <= y_con[con] for sw in range(num_sw) for con in range(num_sw)), name='constr5')
m.addConstrs((gp.quicksum(x_sw_con[(sw, con)] for con in range(num_sw)) == 1 for sw in range(num_sw)), name='constr6')

m.addConstrs(
    (z_c_s_k[((c, s), k)] <= x_sw_con[(s, c)] for s in range(num_sw) for k in range(num_sw) for c in range(num_sw)),
    name='constr8a')
m.addConstrs(
    (z_c_s_k[((c, s), k)] <= x_sw_con[(k, c)] for s in range(num_sw) for k in range(num_sw) for c in range(num_sw)),
    name='constr8b')
m.addConstrs(
    (z_c_s_k[((c, s), k)] == (x_sw_con[(s, c)] + x_sw_con[(k, c)] - 1) for s in range(num_sw) for k in range(num_sw) for
     c in range(num_sw)),
    name='constr8c')


def getS2CDelay(src, dst):
    path = pairs_path_delay_sing[(src, dst)][0]
    sum1 = 0
    for node in path:
        s = getIndex(node)
        # for n in G.neighbors(node):
        #     neighbors.append(getIndex(n))
        # if 0 <= index - 1 <= len(path) - 1:
        #     neighbors.append(getIndex(path[index - 1]))
        # if 0 <= index + 1 <= len(path) - 1:
        #     neighbors.append(getIndex(path[index + 1]))
        if node == src:
            sum1 += sum(2 * x_sw_con[(s, c)] * pairs_delay[(s, c)] for c in range(num_sw))
        else:
            # neighbors = []
            index = path.index(node)
            k = getIndex(path[index - 1])
            if s == 0 and k == 1:
                print(src, '->', dst)
            sum1 += sum(2 * (x_sw_con[(s, c)] - z_c_s_k[((c, s), k)]) * pairs_delay[(s, c)] for c in range(num_sw))
    return sum1


obj = 0
for pair in pairs_load_sing:
    obj += (pairs_path_delay_sing[pair][1] + getS2CDelay(pair[0], pair[1]))
# print(obj)
# obj = sum([pairs_path_delay_sing[pair] + getS2CDelay(pair[0], pair[1]) for pair in pairs_load_sing])
m.setObjective(obj, GRB.MINIMIZE)
m.write('globe.lp')
m.optimize()

status = m.status
if status == GRB.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')
    sys.exit(0)
if status == GRB.OPTIMAL:
    print('The optimal objective is %g' % m.objVal)
    sys.exit(0)
if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)
    sys.exit(0)

print('The model is infeasible; computing IIS')
removed = []

# Loop until we reduce to a model that can be solved
while True:

    m.computeIIS()
    print('\nThe following constraint cannot be satisfied:')
    for c in m.getConstrs():
        if c.IISConstr:
            print('%s' % c.constrName)
            # Remove a single constraint from the model
            removed.append(str(c.constrName))
            m.remove(c)
            break
    print('')

    m.optimize()
    status = m.status

    if status == GRB.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
        sys.exit(0)
    if status == GRB.OPTIMAL:
        break
    if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)
        sys.exit(0)

print('\nThe following constraints were removed to get a feasible LP:')
print(removed)

# print('The model is infeasible; computing IIS')
# m.computeIIS()
# if m.IISMinimal:
#     print('IIS is minimal\n')
# else:
#     print('IIS is not minimal\n')
# print('\nThe following constraint(s) cannot be satisfied:')
# for c in m.getConstrs():
#     if c.IISConstr:
#         print('%s' % c.constrName)

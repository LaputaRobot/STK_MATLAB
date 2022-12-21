from itertools import product
from math import sqrt

import gurobipy as gp
from gurobipy import GRB

# 参数
customers = [(0, 1.5), (2.5, 1.2)]
facilities = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
setup_cost = [3, 2, 3, 1, 3, 3, 4, 3, 2]
cost_per_mile = 1


def compute_distance(loc1, loc2):
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return sqrt(dx * dx + dy * dy)


# 计算MIP模型的关键参数

num_facilities = len(facilities)
num_customers = len(customers)
cartesian_prod = list(product(range(num_customers), range(num_facilities)))

# 计算运输成本

shipping_cost = {(c, f): cost_per_mile * compute_distance(customers[c], facilities[f]) for c, f in cartesian_prod}
m = gp.Model('facility_location')

select = m.addVars(num_facilities, vtype=GRB.BINARY, name='Select')
assign = m.addVars(cartesian_prod, ub=1, vtype=GRB.CONTINUOUS, name='Assign')
# print(select)
# print(assign)

m.addConstrs((assign[(c, f)] <= select[f] for c, f in cartesian_prod), name='Setup2ship')
m.addConstrs((gp.quicksum(assign[(c, f)] for f in range(num_facilities)) == 1 for c in range(num_customers)),
             name='Demand')
# print(select.prod(setup_cost))
m.setObjective(select.prod(setup_cost) + assign.prod(shipping_cost), GRB.MINIMIZE)

m.optimize()

for facility in select.keys():
    if (abs(select[facility].x) > 1e-6):
        print(f"\n 建立仓库的地址为：{facility + 1}.")

for customer, facility in assign.keys():
    if (abs(assign[customer, facility].x) > 1e-6):
        print(f"\n 超市{customer + 1}从工厂{facility + 1}接受 {round(100 * assign[customer, facility].x, 2)} % of 的需求")

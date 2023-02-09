import json
import numpy as np
import math
try:
    from mgmetis.parmetis import adaptive_repart_kway,part_kway
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    has_mpi = True
except (ImportError, ModuleNotFoundError):
    has_mpi = False

def getIndex(string):
    i = int(string[3])
    j = int(string[4])
    return (i - 1) * 10 + j - i + 1

def get_global_csr(t,dtype=None):
    lines=[]
    vwgt=[]
    xadj=[0]
    adjncy = []
    adjwgt = []
    with open('../MetisTopos/{}'.format(t)) as f:
        lines=f.read().splitlines()
    for line in  lines[1:] :
        nums=line.split(' ')[:-1]
        vwgt.append( int(nums[0]))
        is_node=True
        num_adj=0
        for num in nums[1:]:
            if is_node:
                adjncy.append(int(num)-1)
                num_adj+=1
                is_node=False
            else:
                adjwgt.append(int(num))
                is_node=True
        xadj.append(xadj[-1]+num_adj)
    if dtype is None:
        return  vwgt,xadj,adjncy,adjwgt
    return np.asarray(vwgt, dtype=dtype), np.asarray(xadj, dtype=dtype), np.asarray(adjncy, dtype=dtype), np.asarray(adjwgt, dtype=dtype)

def get_pre_part(t,dtype=None):
    sum_file='/home/ygb/STK_MATLAB/PMetis/metis/src/{}/-{}-sum.json'.format(t,t)
    part={}
    part1={}
    with open(sum_file, 'r') as f:
        result = json.load(f)
        p_name=0
        for p in result['assign']:
            for node in result['assign'][p]['switches']:
                part[getIndex(node)]=p_name
                part1[node]=p_name
            p_name+=1
    pre_part=[]
    for i in range(len(part)):
        pre_part.append(part[i+1])
    if dtype is None:
        return  pre_part
    return np.asarray(pre_part, dtype=dtype)

def get_pre_part1():
    file_name="/home/ygb/ParMETIS/graphs/0.part.8"
    liens=[]
    pre_part=[]
    with open(file_name,'r') as f:
        lines=f.read().splitlines()
    for line in lines:
        pre_part.append(int(line))
    return np.asarray(pre_part, dtype=int)

def split_graph(rank, dtype=None):
    g_vwgt, g_xadj, g_adjncy, g_adjwgt= get_global_csr(t, dtype)
    n = xadj.size // 2
    if rank == 0:
        xadj0 = xadj[: n + 1].copy()
        adjs0 = adjncy[: xadj0[-1]].copy()
        if dtype is None:
            return list(xadj0), (adjs0)
        return np.asarray(xadj0, dtype=dtype), np.asarray(adjs0, dtype=dtype)
    xadj1 = xadj[n:].copy()
    adjs1 = adjncy[xadj1[0] :].copy()
    xadj1 -= xadj1[0]
    if dtype is None:
        return list(xadj1), list(adjs1)
    return np.asarray(xadj1, dtype=dtype), np.asarray(adjs1, dtype=dtype)

def adaptive_repart(t):
    try:
        rank = comm.rank
        # xadj, adjncy = split_graph(rank)
        g_vwgt, g_xadj, g_adjncy, g_adjwgt = get_global_csr(t,int)
        # _, part = part_kway(4, xadj, adjs, comm=comm)
        # pre_part=get_pre_part(t,int)
        pre_part=get_pre_part1()
        print(pre_part,'\n')
        re_ubvec=np.float32(2.0)
        options=np.asarray([1,sum([math.pow(2, i) for i in range(11)]),1,1],int) 
        new_cut, part = adaptive_repart_kway(8, g_xadj, g_adjncy, pre_part, vwgt=g_vwgt, adjwgt=g_adjwgt, ubvec=re_ubvec,itr=1, options=options, comm=comm)   
        print(new_cut)
        # NOTE: collect to all processes
        parts = comm.allgather(part)
        part=parts[0]
        if rank==0:
            print(part,'\n')
        # ubvec=np.float32(1.3)
        # options=np.asarray([1,6,1,0],int) 
        # p_cut, p_part = part_kway(8, g_xadj, g_adjncy,vwgt=g_vwgt,adjwgt=g_adjwgt,ubvec=ubvec,options=options)
        # print(p_cut,'\n',p_part,'\n')
    except BaseException as e:
        import sys
        print(e, file=sys.stderr, flush=True)
        comm.Abort(1)
    # adaptive_repart_kway(8, xadj, adjs, comm=comm)


if __name__ == "__main__":
    adaptive_repart('0')
    # get_pre_part('0')

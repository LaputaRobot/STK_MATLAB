from util import *


if __name__ == '__main__':
    # pass
    getAllLoadChange()
    getLoadChangeFile()
    f=open('initLoad.log','w')
    for i in range(1, 9):
        for j in range(1, 10):
            leo,load = getLoad('{}{}'.format(i, j))
            f.write('{},{}\n'.format(leo,load))
    print([1,2][:-1])

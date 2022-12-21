import math
import os

TABLE = {0: 0, 1: 1.6, 2: 6.4, 3: 16, 4: 32, 5: 95, 6: 191, 7: 239, 8: 318}
MATRIX = [[0] * 24,
          [3, 4, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 0, 4, 3, 3, 3, 3, 3, 0, 1, 1, 1, 1],
          [7, 5, 5, 5, 5, 5, 4, 4, 4, 0, 0, 0, 0, 3, 2, 4, 5, 5, 5, 5, 2, 1, 1, 6],
          [8, 8, 4, 7, 6, 6, 7, 7, 5, 5, 1, 1, 1, 0, 1, 3, 7, 5, 4, 2, 1, 1, 1, 8],
          [6, 6, 7, 7, 7, 7, 7, 6, 1, 0, 0, 0, 0, 1, 0, 0, 3, 5, 3, 1, 0, 0, 0, 6],
          [5, 3, 4, 2, 1, 4, 6, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 5, 3, 0, 0, 3],
          [2, 4, 5, 0, 0, 0, 0, 4, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 7, 5, 0, 0],
          [2, 6, 3, 2, 0, 0, 0, 3, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 4, 0, 0],
          [0, 3, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
          [0] * 24, [0] * 24]


def getIndex(lat, lon):
    """
    根据经纬度，将卫星映射到矩阵内
    :param lat: 卫星纬度
    :param lon: 卫星经度
    :return: 卫星在矩阵的index
    """
    x = math.floor((-lat + 90) / 15)
    if lon >= 0:
        y = math.floor(lon / 15)
    else:
        y = 24 - math.ceil((-lon) / 15)
    return x, y


def getLoad(leo, time):
    f = open('../data/72-loads/srcData/{}.csv'.format(leo), 'r')
    # lines = f.readlines()
    line = f.readlines()[time+1].split(',')
    # line = lines[1].split(',')
    # time = float(line[0])
    lat = float(line[1]) * 180 / math.pi
    lon = float(line[2]) * 180 / math.pi
    x, y = getIndex(lat, lon)
    load = MATRIX[x][y]
    # print(leo,time, load)
    # return x, y
    return load


def getLoadChangeFile():
    files = os.listdir('../72-loads/')[:-1]
    print(files)
    for fileN in files:
        file = open('../72-loads/' + fileN, 'r')
        lastLoad = -1
        lines = file.readlines()
        changeFile = open('../72-loads/change/' + fileN, 'w')
        for line in lines[1:]:
            line = line.split(',')
            time = float(line[0])
            lat = float(line[1]) * 180 / math.pi
            lon = float(line[2]) * 180 / math.pi
            x, y = getIndex(lat, lon)
            load = MATRIX[x][y]
            if lastLoad != load:
                changeFile.write('{},{},{},{}\n'.format(time, lat, lon, load))
            lastLoad = load


def getAllLoadChange():
    # allChangeFile = open('allLoadChange.csv', 'w')
    # files = os.listdir('../72-loads/change/')
    allChangeFile = open('Topo-Load-Change.csv', 'w')
    files = ['allLoadChange.csv', 'allTopoChange.csv']
    print(files)
    all_file_dict = []
    for fileN in files:
        # file = open('../72-loads/change/' + fileN, 'r')
        file = open(fileN, 'r')
        LEO_name = fileN.split('.')[0]
        file_dict = {LEO_name: []}
        lines = file.readlines()
        for line in lines:
            # line = line.split(',')
            # t = float(line[0])
            # load = float(line[-1])
            file_dict[LEO_name].append(line.split(','))
        all_file_dict.append(file_dict)
    while len(all_file_dict) > 0:
        min_time = math.inf
        file_index = 0
        min_index = 0
        key = ''
        min_key = ''
        while file_index < len(all_file_dict):
            key = list(all_file_dict[file_index].keys())[0]
            if float(all_file_dict[file_index][key][0][0]) < min_time:
                min_time = float(all_file_dict[file_index][key][0][0])
                min_index = file_index
                min_key = key
            file_index += 1
        allChangeFile.write(
            '{},{},{}'.format(all_file_dict[min_index][min_key][0][0], all_file_dict[min_index][min_key][0][1],
                              all_file_dict[min_index][min_key][0][2]))
        all_file_dict[min_index][min_key].pop(0)
        if len(all_file_dict[min_index][min_key]) == 0:
            all_file_dict.pop(min_index)


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

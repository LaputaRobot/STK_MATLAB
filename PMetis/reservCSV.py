import os


def transfer(fileName):
    matrix = []
    f = open(fileName, 'r')
    for line in f.readlines():
        matrix.append(line.split(','))
    f.close()
    f = open(fileName, 'w')
    rows = len(matrix[1])
    for i in range(rows - 1):
        newLine = ''
        for j in [1,4]:
            newLine += (matrix[j][i] + ',')
        f.write(newLine)
        f.write('\n')


if __name__ == '__main__':
    files = os.listdir('../matlab_Code/72-d')
    print(files)
    for file in files:
        transfer('../matlab_Code/72-d/' + file)
    # transfer('../72-distance/11-21.csv')

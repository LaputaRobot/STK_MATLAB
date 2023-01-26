import sys
import logging

# AssignScheme = 'SamePlane'
# AssignScheme = 'BalCon'
# AssignScheme = 'Greedy'
# AssignScheme = 'METIS'
AssignScheme = 'PyMETIS'

Rewrite = True
# Rewrite = False

LogDestination = 's,f'
# LogDestination = 's'

# PyMetis参数设置
nparts = 8

# MatchOrder = 'Wei'
# MatchOrder = 'LoadWeiLoad'
# MatchOrder = 'SumWei'
MatchOrder = 'SRC'

# MatchScheme = 'WeiDif'
# MatchScheme = 'WeiLoad'
MatchScheme = 'SRC'

allow_err = sys.float_info.epsilon * 100
un_factor = 1.3
max_allow_bal = 0.005
contiguous = True


# BalCon 算法参数配置
MCS = 3
MSSLS = 24

assignment1 = {'LEO11': ['LEO11', 'LEO12', 'LEO13', 'LEO14', 'LEO15', 'LEO16', 'LEO17', 'LEO18', 'LEO19'],
               'LEO21': ['LEO21', 'LEO22', 'LEO23', 'LEO24', 'LEO25', 'LEO26', 'LEO27', 'LEO28', 'LEO29'],
               'LEO31': ['LEO31', 'LEO32', 'LEO33', 'LEO34', 'LEO35', 'LEO36', 'LEO37', 'LEO38', 'LEO39'],
               'LEO41': ['LEO41', 'LEO42', 'LEO43', 'LEO44', 'LEO45', 'LEO46', 'LEO47', 'LEO48', 'LEO49'],
               'LEO51': ['LEO51', 'LEO52', 'LEO53', 'LEO54', 'LEO55', 'LEO56', 'LEO57', 'LEO58', 'LEO59'],
               'LEO61': ['LEO61', 'LEO62', 'LEO63', 'LEO64', 'LEO65', 'LEO66', 'LEO67', 'LEO68', 'LEO69'],
               'LEO71': ['LEO71', 'LEO72', 'LEO73', 'LEO74', 'LEO75', 'LEO76', 'LEO77', 'LEO78', 'LEO79'],
               'LEO81': ['LEO81', 'LEO82', 'LEO83', 'LEO84', 'LEO85', 'LEO86', 'LEO87', 'LEO88', 'LEO89']}
assignment2 = {'LEO11': ['LEO11', 'LEO12', 'LEO13', 'LEO14', 'LEO15', 'LEO16', 'LEO17', 'LEO18', 'LEO19'],
               'LEO22': ['LEO21', 'LEO22', 'LEO23', 'LEO24', 'LEO25', 'LEO26', 'LEO27', 'LEO28', 'LEO29'],
               'LEO33': ['LEO31', 'LEO32', 'LEO33', 'LEO34', 'LEO35', 'LEO36', 'LEO37', 'LEO38', 'LEO39'],
               'LEO44': ['LEO41', 'LEO42', 'LEO43', 'LEO44', 'LEO45', 'LEO46', 'LEO47', 'LEO48', 'LEO49'],
               'LEO55': ['LEO51', 'LEO52', 'LEO53', 'LEO54', 'LEO55', 'LEO56', 'LEO57', 'LEO58', 'LEO59'],
               'LEO66': ['LEO61', 'LEO62', 'LEO63', 'LEO64', 'LEO65', 'LEO66', 'LEO67', 'LEO68', 'LEO69'],
               'LEO77': ['LEO71', 'LEO72', 'LEO73', 'LEO74', 'LEO75', 'LEO76', 'LEO77', 'LEO78', 'LEO79'],
               'LEO88': ['LEO81', 'LEO82', 'LEO83', 'LEO84', 'LEO85', 'LEO86', 'LEO87', 'LEO88', 'LEO89']}

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

LOG_LEVEL = logging.INFO


def get_logger():
    logger = logging.getLogger('pymetis')
    logger.setLevel(LOG_LEVEL)

    # formatter = logging.Formatter('%(filename)-15s:%(lineno)d - %(funcName)s - %(message)s')
    formatter = logging.Formatter(
        './{filename}", line {lineno}\t |{funcName:<20s}| {message}', style='{')
    # formatter = logging.Formatter('./{filename}", line {lineno}\t | {message}',style='{')

    # 2、创建一个handler，用于写入日志文件
    # fh = logging.FileHandler('test.log')
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def printParameters():
    print("AssignScheme: {:4s}".format(AssignScheme))


log = get_logger()

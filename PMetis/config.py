# AssignScheme = 'SamePlane'
# AssignScheme = 'BalCon'
# AssignScheme = 'Greedy'
AssignScheme = 'METIS'
# AssignScheme = 'PyMETIS'

Rewrite = True
# Rewrite = False

LogDestination = 's,f'
# LogDestination = 's'

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


def printParameters():
    print("AssignScheme: {:4s}".format(AssignScheme))


printParameters()

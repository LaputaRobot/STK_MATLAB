import numpy as np

def test_array():
    x=np.zeros(10)
    y=np.zeros(10)
    x=np.append([x][0])
    print(x)

if __name__ == "__main__":
    test_array()
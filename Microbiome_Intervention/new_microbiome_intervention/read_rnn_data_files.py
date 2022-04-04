import pandas as pd
import numpy as np

def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    X = []
    Y = []
    for i, s_x in enumerate(df.iloc[:,0]):
        X.append([])
        values = []
        s_x = s_x.split(';')
        for s in s_x:
            for val in s.replace("[ ","").replace("]","").replace("]","").replace("[","").split(' '):
                if len(val) > 0:
                    val = float(val)
                    values.append(val)
            X[-1].append(values)

    X = np.array(X)

    for i, s_y in enumerate(df.iloc[:, 1]):
        Y.append([])
        s_y = s_y.split(';')
        for s in s_y:
            for val in s.replace("[ ", "").replace("]", "").replace("]", "").replace("[", "").split(' '):
                if len(val) > 0:
                    Y[-1].append(val)

    Y = np.array(Y)

    return X, Y

if __name__ == '__main__':
    x,y=read_csv('final.csv')

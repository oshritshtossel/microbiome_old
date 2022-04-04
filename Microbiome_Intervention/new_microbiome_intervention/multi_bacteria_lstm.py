import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from LearningMethods.intervention_nn import run_NN
from LearningMethods.intervention_rnn import run_RNN
# from Microbiome_Intervention.new_microbiome_intervention import read_rnn_data_files
import os
from sklearn.utils import shuffle

limit = 50
def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    X = []
    Y = []
    for i, s_x in enumerate(df.iloc[:,0]):
        X.append([])
        s_x = s_x.split(';')
        for s in s_x:
            values = []
            for val in s.replace("[ ","").replace("]","").replace("]","").replace("[","").replace(",","").split(' '):
                if len(val) > 0:
                    val = float(val)
                    values.append(val)
            X[-1].append(values)

    X = np.array(X)

    for i, s_y in enumerate(df.iloc[:, 1]):
        Y.append([])
        s_y = s_y.split(';')
        for s in s_y:
            values = []
            for val in s.replace("[ ", "").replace("]", "").replace("]", "").replace("[", "").replace(",","").split(' '):
                if len(val) > 0:
                    val = float(val)
                    values.append(val)
            Y[-1].append(values)

    cols = [[] for i in range(len(Y[0][0]))]
    for l in Y:
        for j in l:
            k = 0
            for i in j:
                cols[k].append(i)
                k+=1
    cols = np.array(cols).transpose()
    intresting_bakterias = []
    for col in range(cols.shape[1]):
        print(len(np.unique(cols[:,col])))
        if len(np.unique(cols[:,col]))>=4:
            intresting_bakterias.append(col)

    Y = []
    for i, s_y in enumerate(df.iloc[:, 1]):
        Y.append([])
        s_y = s_y.split(';')
        for s in s_y:
            values = []
            k = 0
            for val in s.replace("[ ", "").replace("]", "").replace("]", "").replace("[", "").replace(",","").split(' '):
                if len(val) > 0 and k in intresting_bakterias:
                    val = float(val)
                    values.append(val)
                k+=1
            Y[-1].append(values)
    Y = np.array(Y)
    return X, Y
def lengths_clones(samples):
    d = defaultdict(list)
    for s in samples:
        d[len(s)].append(s)
    result = []
    for n in sorted(d, reverse=True):
        if n > 0:
            clone = d[n]
            clone = np.array(clone)
            result.append(
                torch.Tensor(clone.astype(float)))
    return result

def split_to_batches(samples, batch_size):
    batches = list()
    for i in range(len(samples)):
        batch = torch.split(samples[i], batch_size)
        for j in range(len(batch)):
            batches.append(batch[j])
    return batches

def run_multi_bacteria(csv_path, name):
    results = {}

    structures_short = ["001L200H"]
    reg_short =  [0.01]
    dropout_short = [0.1]
    x, y = read_csv(csv_path)
    x, y = shuffle(x, y, random_state=0)
    x_train = x[:int(0.8 * len(x))]
    x_test = x[int(0.8 * len(x)):]
    y_train = y[:int(0.8 * len(x))]
    y_test = y[int(0.8 * len(x)):]
    x_train = lengths_clones(x_train)
    x_test = lengths_clones(x_test)
    y_train = lengths_clones(y_train)
    y_test = lengths_clones(y_test)
    x_train = split_to_batches(x_train, 16)
    x_test = split_to_batches(x_test, 16)
    y_train = split_to_batches(y_train, 16)
    y_test = split_to_batches(y_test, 16)

    for STRUCTURE in structures_short:
        for LEARNING_RATE in [1e-2]:
            for REGULARIZATION in reg_short:
                for DROPOUT in dropout_short:
                    epochs = 70
                    params = {"STRUCTURE": STRUCTURE,
                              "TRAIN_TEST_SPLIT": 0.8,
                              "EPOCHS": epochs,
                              "LEARNING_RATE": LEARNING_RATE,
                              "OPTIMIZER": "Adam",
                              "REGULARIZATION": REGULARIZATION,
                              "DROPOUT": DROPOUT}
                    print(params)

                    res_map = run_RNN([x_train, x_test], [y_train, y_test], missing_values=None, name=name, folder='../../../PycharmProjects/data_microbiome_in_time/' + name, params=params,
                                                 number_of_samples=0, number_of_time_points=0, number_of_bacteria=0,
                                                 GPU_flag=False, task_id='')
                    if str(params) not in results:
                        results[str(params)] = []
                        results[str(params) + " loss"] = []
                    results[str(params)].append(res_map['TEST']['corr'])
                    results[str(params) + " loss"].append(res_map['TEST']['loss'])

    for key in results:
        results[key] = np.nanmean(results[key])
    with open('m_rnn_' + name + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    return np.nanmean(results[str(params)])

if __name__ == '__main__':
    datasets = {}
    # datasets['saliva'] = ['../../../PycharmProjects/data_microbiome_in_time/saliva/time_series.csv', 'saliva']
    # datasets['gdm'] = ['../../../PycharmProjects/data_microbiome_in_time/gdm/time_series.csv', 'gdm']
    datasets['mucositis'] = ['../../../PycharmProjects/data_microbiome_in_time/mucositis/time_series.csv', 'mucositis']
    datasets['allergy'] = ['../../../PycharmProjects/data_microbiome_in_time/allergy/time_series.csv', 'allergy']
    # datasets['vitamineA'] = ['../../../PycharmProjects/data_microbiome_in_time/vitamineA/time_series.csv', 'vitamineA']
    for dataset in datasets:
        run_multi_bacteria(datasets[dataset][0], datasets[dataset][1])
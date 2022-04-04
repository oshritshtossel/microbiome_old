import pickle
import  torch
from collections import defaultdict
import pandas as pd
import numpy as np
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
            for val in s.replace("[ ", "").replace("]", "").replace("]", "").replace("[", "").split(' '):
                if len(val) > 0:
                    Y[-1].append(val)

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

def run_single_bacteria(folder_path, name):
    results = {}

    structures_short = ["001L050H"]
    reg_short =  [0.1]
    dropout_short = [0.2]
    counter = 0
    for file in os.listdir(folder_path):
        if counter > limit:
            break
        x, y = read_csv(os.path.join(folder_path, file))
        x, y = shuffle(x, y, random_state=0)
        unique = len(np.unique(np.array(y).reshape(-1)))
        if unique >= 4:
            x_train = x[:int(0.8*len(x))]
            x_test = x[int(0.8*len(x)):]
            y_train = y[:int(0.8*len(x))]
            y_test = y[int(0.8*len(x)):]
            x_train = lengths_clones(x_train)
            x_test = lengths_clones(x_test)
            y_train = lengths_clones(y_train)
            y_test = lengths_clones(y_test)
            x_train = split_to_batches(x_train, 16)
            x_test = split_to_batches(x_test, 16)
            y_train = split_to_batches(y_train, 16)
            y_test = split_to_batches(y_test, 16)
            counter+=1

            for STRUCTURE in structures_short:
                for LEARNING_RATE in [1e-3]:
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
                                                         GPU_flag=False, task_id='reg')
                            if str(params) not in results:
                                results[str(params)] = []
                                results[str(params) + " loss"]= []
                            results[str(params)].append(res_map['TEST']['corr'])
                            results[str(params) + " loss"].append(res_map['TEST']['loss'])

    for key in results:
        results[key] = np.nanmean(results[key])
    with open('s_rnn_' + name + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    return np.nanmean(results[str(params)])

if __name__ == '__main__':
    datasets = {}
    datasets['saliva'] = ['../../../PycharmProjects/data_microbiome_in_time/saliva/bacteria_time_series', 'saliva']
    datasets['gdm'] = ['../../../PycharmProjects/data_microbiome_in_time/gdm/bacteria_time_series', 'gdm']
    datasets['mucositis'] = ['../../../PycharmProjects/data_microbiome_in_time/mucositis/bacteria_time_series', 'mucositis']
    datasets['allergy'] = ['../../../PycharmProjects/data_microbiome_in_time/allergy/bacteria_time_series', 'allergy']
    datasets['vitamineA'] = ['../../../PycharmProjects/data_microbiome_in_time/vitamineA/bacteria_time_series', 'vitamineA']
    for dataset in datasets:
        run_single_bacteria(datasets[dataset][0], datasets[dataset][1])
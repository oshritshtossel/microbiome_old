import pickle
import json
import  xgboost as xgb
from  sklearn import  metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import pandas as pd
from LearningMethods.Regression.regression_in_time import check_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy import stats
import  matplotlib.pyplot as plt

def plot_corr(dictonary, models):
    x = [str(m[0].upper() + m[1:]).replace('_',' ') for m in models]
    x[0]='Lin regression'
    datasets = []
    corrs = [[] for i in x]
    for key in dictonary:
        corr_for_dataset = dictonary[key][0]
        k = 0
        for model in models:
            corrs[k].append(corr_for_dataset[model])
            k+=1
        datasets.append(str(key).capitalize())
    corrs = np.array(corrs) # row is a model column is a dataset
    X_axis = np.arange(len(x)) * 5

    plt.rcParams["figure.figsize"] = (18, 12)
    plt.bar(X_axis - 1, corrs[:,0], 0.4, label=datasets[0])
    plt.bar(X_axis - 0.5 , corrs[:,1], 0.4, label=datasets[1])
    plt.bar(X_axis, corrs[:,2], 0.4, label=datasets[2])
    plt.bar(X_axis + 0.5, corrs[:,3], 0.4, label=datasets[3])
    plt.bar(X_axis + 1 , corrs[:,4], 0.4, label=datasets[4])

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(X_axis, x)
    # plt.xlabel("Model", )
    plt.ylabel("Correlation")
    plt.legend(prop={'size': 13})
    plt.savefig('final_plot.pdf')
    plt.show()


if __name__ == "__main__":
    # correct only if linears starts with GDM and ends with saliva
    corr_s_nn = [0.13,0.24,0.31,0.28,0.06]
    mse_s_nn = [0.58,0.41,0.33,0.48,0.74]
    corr_s_rnn = [0.25,0.47,0.29,0.22,0.26]
    mse_s_rnn = [0.56,0.07,0.22,0.38,0.86]
    corr_cnn = [0.513, 0.57, 0.553, 0.514, 0.796]
    with open('d.pickle', 'rb') as handle:
        linears = pickle.load(handle)
    with open('svr.pickle', 'rb') as handle:
        svrs = pickle.load(handle)
    dataset_idx = 0
    for dataset in linears:
        svrs[dataset][0].pop('SVR', None)
        svrs[dataset][1].pop('SVR', None)
        linears[dataset][0].update(svrs[dataset][0])
        linears[dataset][1].update(svrs[dataset][1])
        linears[dataset][0]['S_NN'] = corr_s_nn[dataset_idx]
        linears[dataset][1]['S_NN'] = mse_s_nn[dataset_idx]
        linears[dataset][0]['S_LSTM'] = corr_s_rnn[dataset_idx]
        linears[dataset][1]['S_LSTM'] = mse_s_rnn[dataset_idx]
        linears[dataset][0]['IMIC'] = corr_cnn[dataset_idx]
        dataset_idx+=1

    plot_corr(linears, models=linears['GDM'][0].keys())
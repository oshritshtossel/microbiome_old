import pickle
from time import sleep
import LearningMethods.Regression.nn as nn
import pandas as pd
import numpy as np
import regressors
import  torch
import  matplotlib.pyplot as plt
from LearningMethods.intervention_nn import run_NN

def check_dataset(df, split_by=None):
    ids = list(df.index)
    prevs = []
    prev = '-A' + str(split_by)
    if not split_by:
        for item in ids:
            if item[:-1] != prev:
                prevs.append(prev[:-1])
                if item in prevs:
                    print('data frame in a bad format')
                    raise -1
            prev = item[:-1]
    else:
        for item in ids:
            if item.split(split_by)[0] != prev:
                prevs.append(item.split(split_by)[0])
                if item in prevs:
                    print('data frame in a bad format')
                    raise -1
            prev = item.split(split_by)[0]

def get_xs_ys(df, split_by=None):
    ids = list(df.index)
    df_xs = pd.DataFrame(np.zeros(df.shape[1])).T
    df_xs.columns = df.columns
    df_ys = pd.DataFrame(np.zeros(df.shape[1])).T
    df_ys.columns = df.columns
    prev = '0A' + str(split_by)
    if not split_by:
        for item in ids:
            if item[:-1] == prev[:-1] and ord(item[-1]) == ord(prev[-1]) + 1:
                df_xs = df_xs.append(df.loc[prev, :])
                df_ys = df_ys.append(df.loc[item, :])
            prev = item
    else:
        for item in ids:
            if item.split(split_by)[0] == prev.split(split_by)[0] and ord(item.split(split_by)[-1]) == ord(prev.split(split_by)[-1]) + 1:
                df_xs = df_xs.append(df.loc[prev, :])
                df_ys = df_ys.append(df.loc[item, :])
            prev = item
    df_xs = df_xs.iloc[1:, :]
    df_ys = df_ys.iloc[1:, :]
    df_xs.index = np.arange(df_xs.shape[0])
    df_ys.index = np.arange(df_ys.shape[0])
    return df_xs, df_ys

def regression(df, models=[], split_by=None):
    check_dataset(df, split_by=split_by)
    df_xs, df_ys = get_xs_ys(df, split_by)
    coeeficents = {}
    correlations = {}
    mse_means = {}
    for model in models:
        spearmans = []
        mses = []
        coefs = []
        for bacteria in df_ys.columns:
            target = df_ys[bacteria]
            spearman, mse, coef = regressors.learning_cross_val_loop(model,df_xs, target)
            spearmans.append(spearman)
            mses.append(mse)
            coefs.append(coef)

        # columns = the bacteria we are trying to predict
        # rows = the weight of the bacteria in the predictions
        coeeficents[model] = pd.DataFrame(coefs).T
        coeeficents[model].columns =  df_ys.columns
        coeeficents[model].index =  df_ys.columns


        spearmans = np.array(spearmans)[~np.isnan(np.array(spearmans))]
        correlations[model] = np.mean(spearmans)
        mse_means[model] = np.mean(np.array(mses))
    return correlations, mse_means, coeeficents

def nn_runner(df, split_by=None):
    check_dataset(df, split_by=split_by)
    df_xs, df_ys = get_xs_ys(df, split_by)
    results = []
    params = {"STRUCTURE": "001L2000H",
              "TRAIN_TEST_SPLIT": 0.8,
              "LOSS": "custom_rmse_for_missing_values",
              "EPOCHS": 100,
              "LEARNING_RATE": 0.01,
              "OPTIMIZER": "Adam",
              "REGULARIZATION": 0.01,
              "DROPOUT": 0.1,
              "EARLY_STOP": 0}
    for bacteria in df_ys.columns:
        result = run_NN(np.array(df_xs), np.array(df_ys[bacteria]), None, params, name='exp', folder='.', number_of_samples=8, number_of_time_points=0, number_of_bacteria=df_xs.shape[1],
               save_model=False, person_indexes=None, Loss="custom_rmse_for_missing_values", add_conv_layer=False,
               GPU_flag=False, k_fold=False, task_id="")
        results.append(result)

    return np.mean(results)

def nn_multiple_runner(df, split_by=None):
    check_dataset(df, split_by=split_by)
    df_xs, df_ys = get_xs_ys(df, split_by)
    spearman = nn.main(df_xs, df_ys, cv_size= 5, multiple=True)
    return spearman

if __name__ == "__main__":
    # each dataset is a list: spearman by each model, mean_mse by each model, coef_matrix by each model
    datasets_result = {}

    df = pd.read_csv('../../Datasets/GDM/OTU_merged_General_task.csv')
    df.index = ['-'.join(i.split('-')[:-2]) for i in df['ID']]
    df = df.iloc[:, 1:]
    datasets_result['GDM'] = regression(df, ['Linear_regression', 'Lasso', 'Ridge', 'ARD', 'SVR'])
    print(3)

    df = pd.read_csv('../../Datasets/mucusitis/OTU_merged_General_task.csv')
    df.index = df['Unnamed: 0']
    df = df.iloc[:, 2:]
    datasets_result['mucusitis'] = regression(df, ['Linear_regression', 'Lasso','Ridge', 'ARD', 'SVR'])
    print(0)


    df = pd.read_csv('../../Datasets/Allergy/OTU_merged_General_task_for_reg.csv')
    df.index = df['Unnamed: 0']
    df = df.iloc[:, 1:]
    df = df.dropna(how='all')
    df =  df[~df.index.duplicated(keep='last')]
    datasets_result['Allergy'] = regression(df, ['Linear_regression', 'Lasso', 'Ridge', 'ARD', 'SVR'])
    print(1)

    df = pd.read_csv('../../Datasets/VitamineA/OTU_merged_General_task.csv')
    df.index = df['ID']
    df = df.iloc[:, 1:]
    datasets_result['VitamineA'] = regression(df, ['Linear_regression', 'Lasso','Ridge', 'ARD', 'SVR'])
    print(2)

    with open('d.pickle', 'wb') as handle:
        pickle.dump(datasets_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    regressors.plot_bar_regression_models()


import pickle
import LearningMethods.Regression.nn as nn
import pandas as pd
import numpy as np
import regressors
from LearningMethods.intervention_nn import run_NN
from  LearningMethods.Regression import regression_in_time
if __name__ == "__main__":
    # each dataset is a list: spearman by each model, mean_mse by each model, coef_matrix by each model
    datasets_result = {}

    df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/gdm/OTU_merged_General_task.csv')
    df.index = ['-'.join(i.split('-')[:-2]) for i in df['ID']]
    df = df.iloc[:, 1:]
    df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/gdm/gdm_xs.csv')
    df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/gdm/ys.csv')
    df_xs = df_xs.iloc[:,1:]
    df_ys = df_ys.iloc[:,1:]
    datasets_result['GDM'] = regression_in_time.regression(df, ['SVR','SVR poly', 'SVR rbf','SVR sigmoid'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
    print(3)

    df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/mucositis/OTU_merged_General_task.csv')
    df.index = df['Unnamed: 0']
    df = df.iloc[:, 2:]
    df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/mucositis/mucositis_xs.csv')
    df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/mucositis/ys.csv')
    df_xs = df_xs.iloc[:,1:]
    df_ys = df_ys.iloc[:,1:]
    datasets_result['mucusitis'] = regression_in_time.regression(df, ['SVR','SVR poly', 'SVR rbf','SVR sigmoid'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
    print(0)


    df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/allergy/OTU_merged_General_task_for_reg.csv')
    df.index = df['Unnamed: 0']
    df = df.iloc[:, 1:]
    df = df.dropna(how='all')
    df =  df[~df.index.duplicated(keep='last')]
    df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/allergy/Allergy_xs.csv')
    df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/allergy/ys.csv')
    df_xs = df_xs.iloc[:,1:]
    df_ys = df_ys.iloc[:,1:]
    datasets_result['Allergy'] = regression_in_time.regression(df, ['SVR','SVR poly', 'SVR rbf','SVR sigmoid'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
    print(1)

    df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/vitamineA/OTU_merged_General_task.csv')
    df.index = df['ID']
    df = df.iloc[:, 1:]
    df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/vitamineA/VitamineA_xs.csv')
    df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/vitamineA/ys.csv')
    df_xs = df_xs.iloc[:,1:]
    df_ys = df_ys.iloc[:,1:]
    datasets_result['VitamineA'] = regression_in_time.regression(df, ['SVR','SVR poly', 'SVR rbf','SVR sigmoid'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
    print(2)

    df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/saliva/saliva_full_data.csv')
    df.index = df['ID']
    df = df.iloc[:, 1:]
    df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/saliva/saliva_xs.csv')
    df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/saliva/ys.csv')
    df_xs = df_xs.iloc[:,1:]
    df_ys = df_ys.iloc[:,1:]
    datasets_result['saliva'] = regression_in_time.regression(df, ['SVR','SVR poly', 'SVR rbf','SVR sigmoid'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
    print(2)

    with open('svr.pickle', 'wb') as handle:
        pickle.dump(datasets_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    regressors.plot_bar_regression_models()
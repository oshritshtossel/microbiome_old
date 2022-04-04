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

def xgboost(train_x, train_y, val_x, val_y, task, results_dict):
    n_estimators = [25, 50,]
    max_depths = [8, 15, 25]
    colsample_bytrees = [0.5, 1]
    learning_rate = [0.3, 0.1, 0.03, 0.01]  # [0.009, 0.09, 0.3]
    gamma = [0, 0.6]  # [0.5, 0.9]
    objective = ['binary:logistic']
    train_x.columns = [str(i).replace('_','').replace('[','').replace(']','').replace(';','') for i in train_x.columns]
    val_x.columns = [str(i).replace('_','').replace('[','').replace(']','').replace(';','') for i in train_x.columns]

    max = 0
    for n_estimator in n_estimators:
        for max_depts in max_depths:
            for colsample_bytree in colsample_bytrees:
                for l in learning_rate:
                    for g in gamma:
                        params = {'est': n_estimator,'l':l,'g':g, 'max': max_depts, 'task': task, 'colsample': colsample_bytree}
                        param_string = str(params)
                        if param_string not in results_dict:
                            results_dict[param_string] = []
                        if task == 'reg':
                            model = xgb.XGBClassifier(n_estimators=n_estimator,
                                                          colsample_bytree=colsample_bytree,
                                                          max_depth=max_depts,
                                                          bootstrap=True, learning_rate=l, gamma=g)
                            model.fit(train_x, train_y)
                            predicts = model.predict(val_x)
                            corr = stats.spearmanr(predicts, val_y)[0]
                            results_dict[param_string].append(corr)
    return results_dict



def random_forest(train_x,train_y, val_x, val_y, task, results_dict):
    n_estimators = [10, 25,50]
    min_samples_leafs  = [ 10,20, 50]
    max_depths = [10,30, 50]

    for n_estimator in n_estimators:
        for min_samples_leaf in min_samples_leafs:
            for max_depth in max_depths:
                params = {'est': n_estimator, 'min': min_samples_leaf, 'max': max_depth, 'task': task}
                param_string = str(params)
                if param_string not in results_dict:
                    results_dict[param_string] = []
                if task == 'reg':
                    model = RandomForestRegressor(n_estimators=n_estimator,
                                                      min_samples_leaf= min_samples_leaf,
                                                      max_depth=max_depth,
                                                      bootstrap=True)
                    model.fit(train_x, train_y)
                    predicts = model.predict(val_x)
                    corr = stats.spearmanr(predicts, val_y)[0]
                    results_dict[param_string].append(corr)
                elif task == 'class':
                    model = RandomForestClassifier(n_estimators=n_estimator,
                                                   min_samples_leaf=min_samples_leaf,
                                                   max_depth=max_depth,
                                                   bootstrap=True)
                    model.fit(train_x, train_y)
                    predicts = model.predict(val_x)
                    f1 = f1_score(predicts, val_y)
                    results_dict[param_string].append(f1)
    return results_dict

if __name__ == "__main__":
    df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/allergy/OTU_merged_General_task_for_reg.csv')
    df.index = df['Unnamed: 0']
    df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/allergy/Allergy_xs.csv')
    df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/allergy/Allergy_ys.csv')
    df_xs= df_xs.iloc[:, 1:]
    df_ys= df_ys.iloc[:, 1:]
    df = df.iloc[:, 1:]
    check_dataset(df, split_by=None)

    model = 'xgb'
    i = 0
    X_train_val, X_test, y_train_val, y_test = train_test_split(df_xs, df_ys, test_size = 0.15, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.2, random_state = 42)
    train_index = list(X_train.index)
    val_index = list(X_val.index)
    test_index = list(X_test.index)
    results_dict = {}
    for bacteria in df_ys.columns:
        print(i, ':', len(df_ys.columns))
        target = df_ys[bacteria]
        uniqe = len(np.unique(list(target)))
        if uniqe >= 4 and len(np.unique(target[train_index])) >= 4:
            pass
            if model == 'rf':
                random_forest(df_xs.iloc[train_index,:], target[train_index], df_xs.iloc[val_index,:], target[val_index], 'reg', results_dict)
            elif model == 'xgb':
                xgboost(df_xs.iloc[train_index, :], target[train_index], df_xs.iloc[val_index, :], target[val_index], 'reg', results_dict)
        elif uniqe == 1:
          pass
        i+=1

    with open('res_allergy_'+model+'.pickle', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



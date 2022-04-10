import pickle
import pandas as pd
import numpy as np
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

def get_xs_ys(df, split_by=None, drop_ununique=True):
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
            if item.split(split_by)[0] == prev.split(split_by)[0] and int(item.split(split_by)[-1]) == int(prev.split(split_by)[-1]) + 1:
                df_xs = df_xs.append(df.loc[prev, :])
                df_ys = df_ys.append(df.loc[item, :])
            prev = item
    df_xs = df_xs.iloc[1:, :]
    df_ys = df_ys.iloc[1:, :]
    df_xs.index = np.arange(df_xs.shape[0])
    df_ys.index = np.arange(df_ys.shape[0])
    good_cols = []
    if drop_ununique:
        for c in df_ys.columns:
            if len(np.unique(df_ys[c])) >= 4:
                good_cols.append(c)
        df_ys = df_ys[good_cols]
    return df_xs, df_ys

def regression(df, models=[], split_by=None, df_xs=None, df_ys=None, xsysflag=0, shuffle=False):
    # check_dataset(df, split_by=split_by) # return it!!!!!!!!!
    if not xsysflag:
        df_xs, df_ys = get_xs_ys(df, split_by)
    coeeficents = {}
    correlations = {}
    mse_means = {}
    for model in models:
        spearmans = []
        mses = []
        coefs = []
        i =0
        cols = []
        for bacteria in df_ys.columns:
            print(i, ':', len(df_ys.columns))
            target = df_ys[bacteria]
            if shuffle:
                np.random.shuffle(target)
            uniqe = len(np.unique(list(target)))
            if uniqe >= 4:
                spearman, mse, coef = regressors.learning_cross_val_loop(model, df_xs, target)
                spearmans.append(spearman)
                mses.append(mse)
                coefs.append(coef)
                cols.append(bacteria)
            i += 1

        # columns = the bacteria we are trying to predict
        # rows = the weight of the bacteria in the predictions
        coeeficents[model] = pd.DataFrame(coefs).T
        coeeficents[model].columns =  cols
        coeeficents[model].index =  list(df.columns)


        spearmans = np.array(spearmans)[~np.isnan(np.array(spearmans))]
        correlations[model] = np.mean(spearmans)
        mse_means[model] = np.mean(np.array(mses))
        print(str(mse_means[model]), '   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(correlations[model])
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

# if __name__ == "__main__":
#     # each dataset is a list: spearman by each model, mean_mse by each model, coef_matrix by each model
#     datasets_result = {}
#
#     df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/gdm/OTU_merged_General_task.csv')
#     df.index = ['-'.join(i.split('-')[:-2]) for i in df['ID']]
#     df = df.iloc[:, 1:]
#     df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/gdm/gdm_xs.csv')
#     df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/gdm/ys.csv')
#     df_xs = df_xs.iloc[:,1:]
#     df_ys = df_ys.iloc[:,1:]
#     datasets_result['GDM'] = regression(df, ['Linear_regression', 'Lasso', 'Ridge', 'ARD', 'SVR'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
#     print(3)
#
#     df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/mucositis/OTU_merged_General_task.csv')
#     df.index = df['Unnamed: 0']
#     df = df.iloc[:, 2:]
#     df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/mucusitis/mucusitis_xs.csv')
#     df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/mucusitis/ys.csv')
#     df_xs = df_xs.iloc[:,1:]
#     df_ys = df_ys.iloc[:,1:]
#     datasets_result['mucusitis'] = regression(df, ['Linear_regression', 'Lasso','Ridge', 'ARD', 'SVR'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
#     print(0)
#
#
#     df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/allergy/OTU_merged_General_task_for_reg.csv')
#     df.index = df['Unnamed: 0']
#     df = df.iloc[:, 1:]
#     df = df.dropna(how='all')
#     df =  df[~df.index.duplicated(keep='last')]
#     df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/allergy/Allergy_xs.csv')
#     df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/allergy/ys.csv')
#     df_xs = df_xs.iloc[:,1:]
#     df_ys = df_ys.iloc[:,1:]
#     datasets_result['Allergy'] = regression(df, ['Linear_regression', 'Lasso', 'Ridge', 'ARD', 'SVR'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
#     print(1)
#
#     df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/vitamineA/OTU_merged_General_task.csv')
#     df.index = df['ID']
#     df = df.iloc[:, 1:]
#     df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/vitamineA/VitamineA_xs.csv')
#     df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/vitamineA/ys.csv')
#     df_xs = df_xs.iloc[:,1:]
#     df_ys = df_ys.iloc[:,1:]
#     datasets_result['VitamineA'] = regression(df, ['Linear_regression', 'Lasso','Ridge', 'ARD', 'SVR'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
#     print(2)
#
#     df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/saliva/saliva_full_data.csv')
#     df.index = df['ID']
#     df = df.iloc[:, 1:]
#     df_xs = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/saliva/saliva_xs.csv')
#     df_ys = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/saliva/ys.csv')
#     df_xs = df_xs.iloc[:,1:]
#     df_ys = df_ys.iloc[:,1:]
#     datasets_result['saliva'] = regression(df, ['Linear_regression', 'Lasso','Ridge', 'ARD', 'SVR'], df_xs=df_xs, df_ys=df_ys,xsysflag=1)
#     print(2)
#
#     with open('d.pickle', 'wb') as handle:
#         pickle.dump(datasets_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     regressors.plot_bar_regression_models()

if __name__ == "__main__":
    datasets_result = {}

    df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/GVHD/OTU_merged_General_task.csv')
    df.index = df['ID']
    df = df.iloc[:,1:]
    df_xs, df_ys = get_xs_ys(df, split_by='W')
    df_xs.to_csv('../../../PycharmProjects/data_microbiome_in_time/GVHD/df_xs.csv')
    df_ys.to_csv('../../../PycharmProjects/data_microbiome_in_time/GVHD/ys.csv')
    # datasets_result['GVHD'] = regression(df, ['Linear_regression', 'Lasso', 'Ridge', 'ARD', 'SVR', 'SVR poly', 'SVR poly', 'SVR rbf','SVR sigmoid'], df_xs=df_xs, df_ys=df_ys, xsysflag=1)

    df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/pnas/OTU_merged_General_task.csv')
    df.index = df['ID']
    df = df.iloc[:,1:]
    df_xs, df_ys = get_xs_ys(df, split_by='_')
    df_xs.to_csv('../../../PycharmProjects/data_microbiome_in_time/pnas/df_xs.csv')
    df_ys.to_csv('../../../PycharmProjects/data_microbiome_in_time/pnas/ys.csv')
    # datasets_result['pnas'] = regression(df, ['Linear_regression', 'Lasso', 'Ridge', 'ARD', 'SVR', 'SVR poly', 'SVR poly', 'SVR rbf','SVR sigmoid'], df_xs=df_xs, df_ys=df_ys, xsysflag=1)

    df = pd.read_csv('../../../PycharmProjects/data_microbiome_in_time/diab/OTU_merged_General_task.csv')
    df.index = df['ID']
    df = df.iloc[:,1:]
    df_xs, df_ys = get_xs_ys(df, split_by='_')
    df_xs.to_csv('../../../PycharmProjects/data_microbiome_in_time/diab/df_xs.csv')
    df_ys.to_csv('../../../PycharmProjects/data_microbiome_in_time/diab/ys.csv')
    # datasets_result['diab'] = regression(df, ['Linear_regression', 'Lasso', 'Ridge', 'ARD', 'SVR', 'SVR poly', 'SVR poly', 'SVR rbf','SVR sigmoid'], df_xs=df_xs, df_ys=df_ys, xsysflag=1)

    with open('../../Microbiome_Intervention/new_microbiome_intervention/results/e.pickle', 'wb') as handle:
        pickle.dump(datasets_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

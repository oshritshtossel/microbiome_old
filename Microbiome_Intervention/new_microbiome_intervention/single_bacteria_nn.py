import pandas as pd
import numpy as np
from LearningMethods.intervention_rnn import run_RNN
from LearningMethods.intervention_nn import run_NN
import pickle
def run_single_bacteria(xs_path, ys_path, name):
    df_xs = pd.read_csv(xs_path)
    df_ys = pd.read_csv(ys_path)
    df_xs.index = df_xs.iloc[:,0]
    df_ys.index = df_ys.iloc[:,0]
    df_xs = df_xs.iloc[:,1:]
    df_ys = df_ys.iloc[:,1:]
    results = {}

    structures_short = ["001L100H"]
    structures_long =["001L025H", "001L050H", "001L100H", "001L200H", "002L025H025H", "002L050H050H", "002L100H100H"]
    reg_short =  [1]
    reg_long = [0, 0.001, 0.01, 0.1, 0.5, 1]
    dropout_short = [0.1]
    dropout_long = [0, 0.01, 0.1, 0.2, 0.4]
    for col in df_ys.columns[25:75]:
        y = df_ys[col]
        uniqe = len(np.unique(y))
        if uniqe >= 4:
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

                            res_map = run_NN(df_xs, y, missing_values=None, params=params, name='single_nn_'+str(name), folder='../../../PycharmProjects/data_microbiome_in_time/' + name,
                                             number_of_samples=df_xs.shape[0], number_of_time_points=0, number_of_bacteria=df_xs.shape[1],
                                            save_model=True,  person_indexes=None, Loss = "custom_rmse_for_missing_values", add_conv_layer=False, GPU_flag=False, k_fold=False, task_id="")
                            if str(params) not in results:
                                results[str(params)] = []
                                results[str(params) + " loss"]= []
                            results[str(params)].append(res_map['TEST']['corr'])
                            results[str(params) + " loss"].append(res_map['TEST']['loss'])

        else:
            print(uniqe)
    for key in results:
        results[key] = np.nanmean(results[key])
    with open('s_nn_' + name + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    return np.nanmean(results[str(params)])

if __name__ == '__main__':
    datasets = {}
    # datasets['gdm'] = ['../../../PycharmProjects/data_microbiome_in_time/gdm/gdm_xs.csv','../../../PycharmProjects/data_microbiome_in_time/gdm/gdm_ys.csv', 'gdm']
    # datasets['mucositis'] = ['../../../PycharmProjects/data_microbiome_in_time/mucositis/mucositis_xs.csv','../../../PycharmProjects/data_microbiome_in_time/mucositis/mucositis_ys.csv', 'mucositis']
    # datasets['allergy'] = ['../../../PycharmProjects/data_microbiome_in_time/allergy/Allergy_xs.csv','../../../PycharmProjects/data_microbiome_in_time/allergy/Allergy_ys.csv', 'allergy']
    # datasets['vitamineA'] = ['../../../PycharmProjects/data_microbiome_in_time/vitamineA/VitamineA_xs.csv','../../../PycharmProjects/data_microbiome_in_time/vitamineA/VitamineA_ys.csv', 'vitamineA']
    datasets['pnas'] = ['../../../PycharmProjects/data_microbiome_in_time/pnas/df_xs.csv','../../../PycharmProjects/data_microbiome_in_time/pnas/ys.csv', 'pnas']
    datasets['diab'] = ['../../../PycharmProjects/data_microbiome_in_time/diab/df_xs.csv','../../../PycharmProjects/data_microbiome_in_time/diab/ys.csv', 'diab']
    # datasets['GVHD'] = ['../../../PycharmProjects/data_microbiome_in_time/GVHD/df_xs.csv','../../../PycharmProjects/data_microbiome_in_time/GVHD/ys.csv', 'GVHD']
    for dataset in datasets:
        run_single_bacteria(datasets[dataset][0], datasets[dataset][1], datasets[dataset][2])
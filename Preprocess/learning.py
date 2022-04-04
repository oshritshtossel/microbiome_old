import  pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifierCV
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import scikitplot as skplt
from sklearn.metrics import r2_score


def shuffle_train_test(df, groups_path):
    groups = pd.read_csv(groups_path)
    # if len(groups.columns) == 1 and groups.shape[0] == df.shape[0]:
    #     df['ID'] = groups.loc[:,0]
    if len(groups.columns) == 2 and groups.shape[0] == df.shape[0]:
        dic = dict(zip(groups.iloc[:,0].tolist(), groups.iloc[:,1].tolist()))
        df=df.replace({'ID': dic})
    else:
        print("groups file in bad format")
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2).split(df, groups=df['ID']))

    return df.iloc[train_inds],  df.iloc[test_inds]

def evaluate_model(predicts, y):
    try:
        return  roc_auc_score(y, predicts)
    except:
        return 0


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def logistic_reg(train_x, train_y, test_X, test_y):
    train_x, cv_x, train_y, cv_y = train_test_split(train_x, train_y, test_size = 0.2)
    model = LogisticRegression( random_state=0,solver='liblinear')
    model.fit(train_x, train_y)
    predicts = model.predict_proba(cv_x)
    predicts = predicts[:,1]
    score = evaluate_model(predicts, cv_y)
    return score

def ridge_reg(train_x,train_y, test_X, test_y):
    arr = [True, False]
    max = 0
    params = {}
    train_x, cv_x, train_y, cv_y = train_test_split(train_x, train_y, test_size = 0.2)

    for i in arr:
        clf = RidgeClassifierCV(alphas=np.geomspace(0.01, 1000, num=1), normalize=i)
        clf.fit(train_x, train_y)
        predicts = clf._predict_proba_lr(cv_x)
        predicts = predicts[:, 1]
        score = evaluate_model(predicts, cv_y)
        if score > max:
            max = score
            params = clf.get_params()
    return max, params

def random_forest(train_x,train_y, test_X, test_y):
    n_estimators = 50
    min_samples_split = 2
    max_leaf_nodes = 15
    max_depths = 8
    max_features = "log2"



    model = RandomForestClassifier(n_estimators=n_estimators,
                                      min_samples_split=min_samples_split,
                                      max_leaf_nodes=max_leaf_nodes,
                                      max_features=max_features,
                                      max_depth=max_depths,
                                      bootstrap=True,
                                      verbose=0)
    model.fit(train_x, train_y)
    predicts = model.predict_proba(test_X)
    predicts = predicts[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(test_y, predicts)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc = round(roc_auc, 2)

    f, ax = plt.subplots()
    ax.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc}')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    return roc_auc, f


def xgboost(train_x, train_y, test_x, test_y):
    boosterst = ['gbtree']
    n_estimators = [10, 50, 150]
    max_depths = [3,7, 15, 30]
    colsample_bytrees = [0.5,1]
    learning_rate = [0.9, 0.3, 0.1, 0.03]  # [0.009, 0.09, 0.3]
    gamma = [0, 0.6]  # [0.5, 0.9]
    objective  = ['binary:logistic']

    train_x = train_x.set_axis(np.arange(train_x.shape[1]), axis=1, inplace=False)
    test_x = test_x.set_axis(np.arange(test_x.shape[1]), axis=1, inplace=False)

    xg_clf = xgb.XGBClassifier(tree_method='exact', booster='gbtree', max_depth=7,n_estimators=50,\
                               colsample_bytree=1, learning_rate=0.03,\
                               gamma=0, objective='binary:logistic')
    xg_clf.fit(train_x, train_y)
    predicts = xg_clf.predict_proba(test_x)
    predicts = predicts[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(test_y, predicts)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)
    roc_auc = round(roc_auc, 2)

    # f, ax = plt.subplots()
    # test_y = test_y.to_numpy()
    # ax.plot(skplt.metrics.plot_roc(test_y, predicts))

    f, ax = plt.subplots()
    ax.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc}')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    return roc_auc, f



def fc_network(train_x,train_y, test_x, test_y):
    # train_x, cv_x, train_y, cv_y = train_test_split(train_x, train_y, test_size = 0.2)

    # activations = ['identity', 'relu', 'logistic', 'tanh']
    # solvers = ['lbfgs', 'adam']
    # alphas = [0.00001,0.0001, 0.001, 0.01, 0.1]
    # hidden_sizes = [(8),(12),(8,8), (16,16), (32,32)]
    #
    #
    # clf = MLPClassifier(activation='relu', solver='adam', alpha=1, hidden_layer_sizes=(16))
    # clf.fit(train_x, train_y)
    # predicts = clf.predict_proba(test_x)
    # predicts = predicts[:, 1]
    # auc = evaluate_model(predicts, test_y)
    # fpr, tpr, threshold = metrics.roc_curve(test_y, predicts)
    # roc_auc = metrics.auc(fpr, tpr)
    #
    # f.title('Receiver Operating Characteristic')
    # f.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # f.legend(loc='lower right')
    # f.plot([0, 1], [0, 1], 'r--')
    # f.xlim([0, 1])
    # f.ylim([0, 1])
    # f.ylabel('True Positive Rate')
    # f.xlabel('False Positive Rate')
    # ax = f.gca()
    # print(auc)
    # return auc, ax




    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 8, train_x.shape[1], 8, 1

    train_ds = TensorDataset(torch.tensor(train_x.to_numpy()).float(), torch.tensor(train_y.to_numpy()).float())
    test_ds = TensorDataset(torch.tensor(test_x.to_numpy()).float(), torch.tensor(test_y.to_numpy()).float())
    train_dl = DataLoader(train_ds, batch_size=8)
    test_dl = DataLoader(test_ds, batch_size=8)


    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(1,500):
        epoch_acc = 0
        epoch_loss = 0
        for xb, yb in train_dl:
            model.train()
            pred = torch.sigmoid(model(xb))
            loss = loss_fn(pred, yb.unsqueeze(1))
            acc =  binary_acc(pred, yb.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            print(
                f'Epoch {t + 0:03}: | Loss: {epoch_loss / len(train_dl):.5f} | Acc: {epoch_acc / len(train_dl):.3f}')

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch, yb in test_dl:
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    flat_y_list = [item for sublist in y_pred_list for item in sublist]
    fpr, tpr, threshold = metrics.roc_curve(test_y, flat_y_list)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc = round(roc_auc, 2)


    f, ax = plt.subplots()
    ax.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc}')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    return roc_auc, f


def prepare_for_learn(df_x, df_y, groups, ip):
    df_x = df_x.sort_values(by=['ID'])
    df_y = df_y.sort_values(by=['ID'])
    df_x = shuffle(df_x, random_state=4)
    df_y = shuffle(df_y, random_state=4)
    if groups:
        train_inds, test_inds = shuffle_train_test(df_x, ip + "/GROUPS.csv")
        df_y = df_y['Tag']
        train_inds = train_inds.iloc[:, 1:]
        test_inds = test_inds.iloc[:,1:]
        train_y = df_y.iloc[train_inds.index]
        test_y = df_y.iloc[test_inds.index]
        return train_inds, train_y, test_inds, test_y
    else:
        print(df_y.columns)
        print(df_y.iloc[:,1])
        df_y = df_y['Tag']
        df_x = df_x.iloc[:, 1:]
        train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size = 0.2)
    return train_x, train_y, test_x, test_y

def plot_scatter_plot(real_y, predicted_y, fig, ax):
    fig.set_size_inches(16, 12)
    ax.scatter(real_y, predicted_y, 15)
    plt.xlabel('real_y')
    plt.ylabel('predicted_y')
    plt.title('scatter plot')
    xlim = plt.xlim()[1]
    ylim = plt.ylim()[1]
    plt.plot(np.array([0, xlim]), np.array([0,ylim]))
    plt.grid(True)
    return fig

def random_forest_reg(train_x,train_y, test_X, test_y):
    n_estimators = 50
    min_samples_split = 2
    max_leaf_nodes = 15
    max_depths = 8
    max_features = "log2"



    model = RandomForestRegressor(n_estimators=n_estimators,
                                      min_samples_split=min_samples_split,
                                      max_leaf_nodes=max_leaf_nodes,
                                      max_features=max_features,
                                      max_depth=max_depths,
                                      bootstrap=True,
                                      verbose=0)
    model.fit(train_x, train_y)
    predicts = model.predict(test_X)
    r2 = r2_score(test_y, predicts)
    fig, ax = plt.subplots()
    fig = plot_scatter_plot(test_y, predicts, fig, ax)
    print(r2)

    return r2, fig

def fc_network_reg(train_x,train_y, test_x, test_y):
    N, D_in, H, D_out = 8, train_x.shape[1], 8, 1

    train_ds = TensorDataset(torch.tensor(train_x.to_numpy()).float(), torch.tensor(train_y.to_numpy()).float())
    test_ds = TensorDataset(torch.tensor(test_x.to_numpy()).float(), torch.tensor(test_y.to_numpy()).float())
    train_dl = DataLoader(train_ds, batch_size=8)
    test_dl = DataLoader(test_ds, batch_size=8)

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

    loss_fn = torch.nn.MSELoss()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(1, 500):
        epoch_acc = 0
        epoch_loss = 0
        for xb, yb in train_dl:
            model.train()
            pred = model(xb)
            loss = loss_fn(pred, yb.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(
                f'Epoch {t + 0:03}: | Loss: {epoch_loss / len(train_dl):.5f}')

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch, yb in test_dl:
            y_test_pred = model(X_batch)
            y_test_pred = y_test_pred
            y_pred_list.append(y_test_pred.numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    flat_y_list = [item for sublist in y_pred_list for item in sublist]
    r2 = r2_score(test_y, flat_y_list)
    fig, ax = plt.subplots()
    fig = plot_scatter_plot(test_y, flat_y_list, fig, ax)
    print(r2)

    return r2, fig


def xgboost_reg(train_x,train_y, test_x, test_y):

    objective = ['reg:squarederror']

    train_x = train_x.set_axis(np.arange(train_x.shape[1]), axis=1, inplace=False)
    test_x = test_x.set_axis(np.arange(test_x.shape[1]), axis=1, inplace=False)

    xg_clf = xgb.XGBClassifier(tree_method='exact', booster='gbtree', max_depth=7, n_estimators=50, \
                               colsample_bytree=1, learning_rate=0.03, \
                               gamma=0, objective=objective)
    xg_clf.fit(train_x, train_y)
    predicts = xg_clf.predict(test_x)
    r2 = r2_score(test_y, predicts)
    fig, ax = plt.subplots()
    fig = plot_scatter_plot(test_y, predicts, fig, ax)
    print(r2)

    return r2, fig

if __name__ == "__main__":
    # train_x = pd.read_csv("/home/roy/Downloads/split_datasets/Allergy_nut_split_dataset/train_val_set_nut_microbiome.csv")
    # train_y = pd.read_csv("/home/roy/Downloads/split_datasets/Allergy_nut_split_dataset/train_val_set_nut_tags.csv")
    # train_x = train_x.sort_values(by=['ID'])
    # train_y = train_y.sort_values(by=['ID'])
    # train_x = shuffle(train_x, random_state=1)
    # train_y = shuffle(train_y, random_state=1)
    # train_x = train_x.iloc[:,1:]
    # train_y = train_y.iloc[:,1]
    #
    # test_x = pd.read_csv("/home/roy/Downloads/split_datasets/Allergy_nut_split_dataset/test_set_nut_microbiome.csv")
    # test_y = pd.read_csv("/home/roy/Downloads/split_datasets/Allergy_nut_split_dataset/test_set_nut_tags.csv")
    # test_x = test_x.sort_values(by=['ID'])
    # test_y = test_y.sort_values(by=['ID'])
    # test_x = shuffle(test_x, random_state=1)
    # test_y = shuffle(test_y, random_state=1)
    # test_x = test_x.iloc[:,1:]
    # test_y = test_y.iloc[:,1]
    #
    # print(fc_network(train_x, train_y, test_x, test_y))

    train_x = pd.read_csv("/home/roy/Downloads/split_datasets/Nugent_split_dataset/train_val_set_nugent_microbiome.csv")
    train_y = pd.read_csv("/home/roy/Downloads/split_datasets/Nugent_split_dataset/train_val_set_nugent_tags.csv")
    train_x = train_x.sort_values(by=['ID'])
    train_y = train_y.sort_values(by=['ID'])
    train_x = shuffle(train_x, random_state=4)
    train_y = shuffle(train_y, random_state=4)
    train_x = train_x.iloc[:,1:]
    train_y = train_y.iloc[:,1]

    test_x = pd.read_csv("/home/roy/Downloads/split_datasets/Nugent_split_dataset/test_set_nugent_microbiome.csv")
    test_y = pd.read_csv("/home/roy/Downloads/split_datasets/Nugent_split_dataset/test_set_nugent_tags.csv")
    test_x = test_x.sort_values(by=['ID'])
    test_y = test_y.sort_values(by=['ID'])
    test_x = shuffle(test_x, random_state=4)
    test_y = shuffle(test_y, random_state=4)
    test_x = test_x.iloc[:,1:]
    test_y = test_y.iloc[:,1]

    print(xgboost(train_x, train_y, test_x, test_y))

    # train_x = pd.read_csv("/home/roy/Downloads/split_datasets/Black_vs_White_split_dataset/train_val_set_Black_vs_White_microbiome.csv")
    # train_y = pd.read_csv("/home/roy/Downloads/split_datasets/Black_vs_White_split_dataset/train_val_set_Black_vs_White_tags.csv")
    # train_x = train_x.sort_values(by=['ID'])
    # train_y = train_y.sort_values(by=['ID'])
    # train_x = shuffle(train_x, random_state=4)
    # train_y = shuffle(train_y, random_state=4)
    # train_x = train_x.iloc[:, 1:]
    # train_y = train_y.iloc[:, 1]
    #
    # test_x = pd.read_csv("/home/roy/Downloads/split_datasets/Black_vs_White_split_dataset/test_set_Black_vs_White_microbiome.csv")
    # test_y = pd.read_csv("/home/roy/Downloads/split_datasets/Black_vs_White_split_dataset/test_set_Black_vs_White_tags.csv")
    # test_x = test_x.sort_values(by=['ID'])
    # test_y = test_y.sort_values(by=['ID'])
    # test_x = shuffle(test_x, random_state=4)
    # test_y = shuffle(test_y, random_state=4)
    # test_x = test_x.iloc[:, 1:]
    # test_y = test_y.iloc[:, 1]
    #
    # print(fc_network(train_x, train_y, test_x, test_y))




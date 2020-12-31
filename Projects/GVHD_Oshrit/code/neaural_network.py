import random
from typing import Tuple, Any
from warnings import warn
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from torch import optim

from torch.utils.data import TensorDataset, DataLoader

from Projects.GVHD_Oshrit.code.nn_models import Model_luzon
from Projects.GVHD_Oshrit.code.plotting_tools import plot_and_save_losses, create_heatmap, scatter_pred_vs_tag
from Projects.GVHD_Oshrit.code.preprocces import delete_features_according_to_key

NUM_OF_RUNS_WITH_NO_CHANGE_IN_LOSS = 20


def split_by_id(Data, Class, ID, test_size=0.33) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    split data to train and test keeping the samples of the same person to be together in the test or in the train
    :param Data: features df
    :param Class: tag df
    :param ID: column to divide and put samples with the same data of it together
    :param test_size: the fraction of test from the data
    """
    unique_id = ID.unique()
    num_id = unique_id.size
    unique_id_permute = np.random.permutation(unique_id)
    len_test = round(num_id * test_size)
    test_id = unique_id_permute[0:len_test]
    train_id = unique_id_permute[len_test:num_id]
    train_id_flag = ID.isin(train_id)
    test_id_flag = ID.isin(test_id)
    y_train = Class[train_id_flag]
    y_test = Class[test_id_flag]
    X_train = Data[train_id_flag]
    X_test = Data[test_id_flag]
    return X_train, X_test, y_train, y_test, ID[train_id_flag]


def with_censored_split_train_test_censored_and_uncensord(censor_df: pd.DataFrame, uncensor_df: pd.DataFrame,
                                                          tag_name: str,
                                                          id_col="subjid", additional_cols_to_keep: list = [],
                                                          train_percent=0.7, bacteria_col_keyword="k__Bacteria") -> \
Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    REMARK: Works for timedelta y type only
    1.split the data to train which consist of all the censored samples and completes to train percent (70) of
    the whole data by uncensored samples. The split keeps that samples of the same person should be together in the test
    together in the train
    2. creates a flag column of censored or uncensored for the loss function 1 = censored, 0 = uncensored
    3. deletes all the features which are not bacterias from the  x train and x test
    4. prepares the test df with the proper tag column
    5. makes all dfs to tensors.
    returns- train data and test data as tensors

    :param censor_df: data frame of censored data with the data augmentation inside
    :param uncensor_df: data frame of uncensored data
    :param tag_name: name of tag
    :param train_percent: size of train
    :param bacteria_col_keyword: words of the columns we want to remain
    :return:

    Usage example:
    train_data, val_dataset, index_of_additional_parameters = split_train_test_censored_and_uncensord(censor_df,
                                                                                                      uncensor_df,
                                                                                                      tag_name)
    """
    additional_cols_to_keep.append("is_censored")
    if len(censor_df.columns) != len(uncensor_df.columns):
        warn(
            "Censored and uncensored dataframes have different number of columns."
            " Adding a 0 column in place of the missing columns")
        diffs = set(censor_df.columns).difference(uncensor_df.columns)
        for diff in diffs:
            if diff in censor_df:
                uncensor_df[diff] = 0
            else:
                censor_df[diff] = 0

    # Censored train
    censored_train_df = censor_df
    # add the flag column with censoring information
    censored_train_df["is_censored"] = 1

    x_train_df_from_uncensored, x_test_df, y_train_df, y_test_df, _ = split_by_id(
        uncensor_df, uncensor_df[tag_name], uncensor_df[id_col], test_size=0.3)

    # add the flag column with censoring information
    x_train_df_from_uncensored["is_censored"] = 0
    pre_x_train_only_uncensord = delete_features_according_to_key(x_train_df_from_uncensored, additional_cols_to_keep)
    pre_x_train_only_uncensord[tag_name] = x_train_df_from_uncensored[tag_name]

    # Test is always uncensored
    x_test_df["is_censored"] = 0

    x_train_only_censord, x_censored_test_df, y_censored_train_df, y_censored_test_df, _ = split_by_id(
        censored_train_df, censored_train_df[tag_name], censored_train_df[id_col],
        test_size=0.3)
    pre_x_train_only_censord = delete_features_according_to_key(x_train_only_censord, additional_cols_to_keep)
    pre_x_train_only_censord[tag_name] = censored_train_df[tag_name]

    # only microbiom features:
    x_test_df = delete_features_according_to_key(x_test_df, keep=["is_censored"],
                                                 bacteria_col_keyword=bacteria_col_keyword)
    x_censored_test_df = delete_features_according_to_key(x_censored_test_df, additional_cols_to_keep)
    x_censored_test_df = delete_features_according_to_key(x_censored_test_df, keep=["is_censored"],
                                                          bacteria_col_keyword=bacteria_col_keyword)
    x_test_df = x_test_df.append(x_censored_test_df)
    y_test_df = y_test_df.append(y_censored_test_df)

    x_train_df = pre_x_train_only_uncensord.append(pre_x_train_only_censord)
    y_train_df = y_train_df.append(y_censored_train_df)
    ######################################################################################################################################################
    x_train_df[id_col] = censor_df[id_col]
    x_train_df[id_col].update(uncensor_df[id_col])

    return x_train_df, y_train_df, x_test_df, y_test_df, -len(additional_cols_to_keep)


def split_train_test_censored_and_uncensord(censor_df: pd.DataFrame, uncensor_df: pd.DataFrame, tag_name: str,
                                            id_col="subjid", additional_cols_to_keep: list = [],
                                            train_percent=0.7, bacteria_col_keyword="k__Bacteria") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    REMARK: Works for timedelta y type only
    1.split the data to train which consist of all the censored samples and completes to train percent (70) of
    the whole data by uncensored samples. The split keeps that samples of the same person should be together in the test
    together in the train
    2. creates a flag column of censored or uncensored for the loss function 1 = censored, 0 = uncensored
    3. deletes all the features which are not bacterias from the  x train and x test
    4. prepares the test df with the proper tag column
    5. makes all dfs to tensors.
    returns- train data and test data as tensors

    :param censor_df: data frame of censored data with the data augmentation inside
    :param uncensor_df: data frame of uncensored data
    :param tag_name: name of tag
    :param train_percent: size of train
    :param bacteria_col_keyword: words of the columns we want to remain
    :return:

    Usage example:
    train_data, val_dataset, index_of_additional_parameters = split_train_test_censored_and_uncensord(censor_df,
                                                                                                      uncensor_df,
                                                                                                      tag_name)
    """
    additional_cols_to_keep.append("is_censored")
    if len(censor_df.columns) != len(uncensor_df.columns):
        warn(
            "Censored and uncensored dataframes have different number of columns."
            " Adding a 0 column in place of the missing columns")
        diffs = set(censor_df.columns).difference(uncensor_df.columns)
        for diff in diffs:
            if diff in censor_df:
                uncensor_df[diff] = 0
            else:
                censor_df[diff] = 0

    # Censored train
    censored_train_df = censor_df
    # add the flag column with censoring information
    censored_train_df["is_censored"] = 1
    pre_x_train_only_censord = delete_features_according_to_key(censored_train_df, additional_cols_to_keep)
    pre_x_train_only_censord[tag_name] = censored_train_df[tag_name]

    x_train_df_from_uncensored, x_test_df, y_train_df, y_test_df, _ = split_by_id(
        uncensor_df, uncensor_df[tag_name], uncensor_df[id_col], test_size=0.3)

    # add the flag column with censoring information
    x_train_df_from_uncensored["is_censored"] = 0
    pre_x_train_only_uncensord = delete_features_according_to_key(x_train_df_from_uncensored, additional_cols_to_keep)
    pre_x_train_only_uncensord[tag_name] = x_train_df_from_uncensored[tag_name]

    # Test is always uncensored
    x_test_df["is_censored"] = 0
    # only microbiom features:
    x_test_df = delete_features_according_to_key(x_test_df, keep=["is_censored"],
                                                 bacteria_col_keyword=bacteria_col_keyword)

    x_train_df = pre_x_train_only_censord.append(pre_x_train_only_uncensord)
    y_train_df = y_train_df.append(censored_train_df[tag_name])
    ######################################################################################################################################################
    x_train_df[id_col] = censor_df[id_col]
    x_train_df[id_col].update(uncensor_df[id_col])

    return x_train_df, y_train_df, x_test_df, y_test_df, -len(additional_cols_to_keep)


def make_train_step_no_last_columns(model, loss_fn, optimizer, index_of_last_col_to_learn):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x[:, :index_of_last_col_to_learn])
        # Computes loss
        loss = loss_fn(y, yhat, x[:, index_of_last_col_to_learn:])
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item(), yhat

    # Returns the function that will be called inside the train loop
    return train_step


def My_loss(alpha, beta):
    alpha = torch.tensor([float(alpha)], requires_grad=True)
    beta = torch.tensor([float(beta)], requires_grad=True)

    def my_loss(y, yhat, properties, val=False):
        """
        calculates the loss of the model
        :param properties: [0] - , [1] - is censored
        :param y: real time
        :param yhat: predicted time
        :param is_censored: flag to know if the sample is censored or uncensored
        :return: the total loss
        """
        loss = 0
        for i in range(len(y)):
            if val:
                censored = properties[i]
            else:
                censored = properties[i][-1]

            if censored == 0:
                # Uncensored - mse
                loss += (y[i] - yhat[i]) ** 2
            else:
                # Censored - test doesn't enter this scope
                loss += alpha * ((y[i] - yhat[i]) ** 2) + beta * (max(0, (properties[i][0] - yhat[i])) ** 2)
        if loss == 0:
            return torch.tensor([0.0], requires_grad=True)
        return loss / len(y)

    return my_loss


def try_split_train_validation(x_train: pd.DataFrame, y_train: pd.Series, tag_name, k_fold: int = 5, id_col="subjid") -> \
        Tuple[
            DataLoader, Any]:
    """
    validation comtain censored and uncemsored splitting together
    so the number of crnsored and uncensored may change
    1.split the train to 2 subgroups of train and validation having an opportunity to k fold
     while samples of the same person would be together in the validation or together in the train
     2. make the dfs to tensor floats
    @param x_train: df of features of the train and validation together
    @param y_train: df of tags of train and validation together
    @param tag_name: name of tag column
    @param k_fold: int with the number of kfold needed if kfold is 0, there is no kfold
    @param id_col: subjid column
    @return: train tensor validation tensor
    """
    if k_fold == 0:
        train_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(x_train.values).float(),
                                                        torch.from_numpy(y_train.values).float()), batch_size=20)
        return train_loader, None

    X_train, X_valid, y_train, y_valid, _ = split_by_id(x_train, x_train[tag_name],
                                                        x_train[id_col],
                                                        test_size=1 / k_fold)

    del X_train[tag_name]
    del X_train[id_col]
    del X_valid[tag_name]
    del X_valid[id_col]
    valid_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(X_valid.to_numpy(dtype="float")).float(),
                                                    torch.from_numpy(y_valid.to_numpy(dtype="float")).float()),
                              batch_size=20)
    train_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(X_train.to_numpy(dtype="float")).float(),
                                                    torch.from_numpy(y_train.to_numpy(dtype="float")).float()),
                              batch_size=20)
    return train_loader, valid_loader


def split_train_validation(x_train: pd.DataFrame, y_train: pd.Series, tag_name, k_fold: int = 5, id_col="subjid") -> \
        Tuple[
            DataLoader, Any]:
    """
    1.split the train to 2 subgroups of train and validation having an opportunity to k fold
     while samples of the same person would be together in the validation or together in the train
     2. make the dfs to tensor floats
    @param x_train: df of features of the train and validation together
    @param y_train: df of tags of train and validation together
    @param tag_name: name of tag column
    @param k_fold: int with the number of kfold needed if kfold is 0, there is no kfold
    @param id_col: subjid column
    @return: train tensor validation tensor
    """
    if k_fold == 0:
        train_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(x_train.values).float(),
                                                        torch.from_numpy(y_train.values).float()), batch_size=20)
        return train_loader, None
    uncensored_x_train = x_train.loc[x_train["is_censored"] == 0]
    censored_x_train = x_train.loc[x_train["is_censored"] == 1]
    X_train, X_valid, y_train, y_valid, _ = split_by_id(uncensored_x_train, uncensored_x_train[tag_name],
                                                        uncensored_x_train[id_col],
                                                        test_size=1 / k_fold)
    X_train = X_train.append(censored_x_train)
    y_train = X_train[tag_name]
    del X_train[tag_name]
    del X_train[id_col]
    del X_valid[tag_name]
    del X_valid[id_col]
    valid_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(X_valid.to_numpy(dtype="float")).float(),
                                                    torch.from_numpy(y_valid.to_numpy(dtype="float")).float()),
                              batch_size=20)
    train_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(X_train.to_numpy(dtype="float")).float(),
                                                    torch.from_numpy(y_train.to_numpy(dtype="float")).float()),
                              batch_size=20)
    return train_loader, valid_loader


def with_censor_split_train_validation(x_train: pd.DataFrame, y_train: pd.Series, tag_name, k_fold: int = 5,
                                       id_col="subjid") -> \
        Tuple[
            DataLoader, Any]:
    """
    1.split the train to 2 subgroups of train and validation having an opportunity to k fold
     while samples of the same person would be together in the validation or together in the train
     2. make the dfs to tensor floats
    @param x_train: df of features of the train and validation together
    @param y_train: df of tags of train and validation together
    @param tag_name: name of tag column
    @param k_fold: int with the number of kfold needed if kfold is 0, there is no kfold
    @param id_col: subjid column
    @return: train tensor validation tensor
    """
    if k_fold == 0:
        train_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(x_train.values).float(),
                                                        torch.from_numpy(y_train.values).float()), batch_size=20)
        return train_loader, None
    uncensored_x_train = x_train.loc[x_train["is_censored"] == 0]
    censored_x_train = x_train.loc[x_train["is_censored"] == 1]
    # split the uncensored samples to train and valid
    X_train, X_valid, y_train, y_valid, _ = split_by_id(uncensored_x_train, uncensored_x_train[tag_name],
                                                        uncensored_x_train[id_col],
                                                        test_size=1 / k_fold)

    # split the censored samples to train and valid
    X_train_n_c, X_valid_n_c, y_train_n_c, y_valid_n_c, _ = split_by_id(censored_x_train, censored_x_train[tag_name],
                                                                        censored_x_train[id_col],
                                                                        test_size=1 / k_fold)
    X_train = X_train.append(X_train_n_c)
    X_valid = X_valid.append(X_valid_n_c)
    y_train = X_train[tag_name]
    y_valid = X_valid[tag_name]
    del X_train[tag_name]
    del X_train[id_col]
    del X_valid[tag_name]
    del X_valid[id_col]
    valid_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(X_valid.to_numpy(dtype="float")).float(),
                                                    torch.from_numpy(y_valid.to_numpy(dtype="float")).float()),
                              batch_size=20)
    train_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(X_train.to_numpy(dtype="float")).float(),
                                                    torch.from_numpy(y_train.to_numpy(dtype="float")).float()),
                              batch_size=20)
    return train_loader, valid_loader


class custom_nn_runner(object):
    def __init__(self, model, loss_function, optimizer: str = "Adam", learning_rate=0.1, weight_decay=0.1
                 , k_fold=0, epochs=100):
        self.model_kind = model
        self.loss_fn = loss_function
        self.k_fold = k_fold
        self.max_epochs = epochs
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

    def _train(self, train_loader, train_step):
        """
        the training process
        @param train_loader:
        @param train_step:
        @return: loss of train
        """
        # Train
        total_loss = 0
        num_of_iters_train = 0
        Yhat, Y_gt = np.array([]), []
        for x_batch, y_batch in train_loader:
            num_of_iters_train += 1
            # Performs one train step and returns the corresponding loss
            loss, yhat = train_step(x_batch, y_batch)
            Yhat = np.append(Yhat, yhat.detach().numpy().T)
            Y_gt.extend(y_batch.numpy())
            total_loss += loss
        return total_loss, Yhat, Y_gt

    def _test(self, loader, index_to_not_test_from):
        """
        calculate the loss and mse loss of the test or validation
        @param loader:
        @param index_to_not_test_from:
        @return: loss and mse_loss
        """
        # Test evaluation
        with torch.no_grad():
            num_of_iters_test = 0
            loss = 0
            mse_loss = 0
            all_preds, all_gts = [], []
            Yhat, Y = np.array([]), np.array([])
            for x_val, y_val in loader:
                num_of_iters_test += 1
                self.model.eval()

                yhat = self.model(x_val[:, :index_to_not_test_from])
                loss += self.loss_fn(y_val, yhat, x_val[:, index_to_not_test_from:])
                mse_loss += self.mse(y_val, yhat, x_val[:, index_to_not_test_from:])
                all_preds.extend(yhat.numpy())
                all_gts.extend(y_val.numpy())
                Yhat = np.append(Yhat, yhat.numpy())
                Y = np.append(Y, y_val.numpy())
            corr = spearmanr(all_preds, all_gts)[0]
            r2 = r2_score(all_gts, all_preds)
            # plt.scatter(all_preds,all_gts)
            # plt.show()
        return loss, mse_loss, corr, r2, Yhat, Y

    def mse(self, y, yhat, properties, val=False):
        """
        loss of only the censored samples- regular mse
        @param y: real tag
        @param yhat: predicted tag
        @param properties:
        @param val:
        @return: mse loss value
        """
        loss = 0
        for i in range(len(y)):
            if val:
                censored = properties[i]
            else:
                censored = properties[i][-1]

            if censored == 0:
                # Uncensored - mse
                loss += (y[i] - yhat[i]) ** 2
        return loss / len(y)

    def run_train(self, x_train_df, y_train_df, index_of_additional_parameters, test_loader, tag_name):
        """
        1. split to test and then split the train to train + validation
        2. init the model with the right parameters
        3. create train step, loss and optimizer
        4. runs 1000 epochs or till convergence
        @param censor_df: df of censored samples
        @param uncensor_df: df of uncensored samples
        @param tag_name: name of column of tag
        @return: list of train losses, list of valid losses, list of tests losses,
        mse value of last validation epoch
        """
        # Spilt to train test
        train_loader, valid_loader = split_train_validation(x_train_df, y_train_df, tag_name, k_fold=self.k_fold)

        # Init the model with number of features
        self.model = self.model_kind(len(train_loader.dataset[0][0]) + index_of_additional_parameters)
        if self.optimizer.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), weight_decay=self.weight_decay, lr=self.learning_rate)

        # Creates the train_step function for our model, loss function and optimizer
        train_step = make_train_step_no_last_columns(self.model, self.loss_fn, optimizer,
                                                     index_of_additional_parameters)
        # Pre operations to train
        train_losses, test_losses, valid_losses = [], [], []
        min_loss = float("inf")
        num_of_runs_with_no_change = NUM_OF_RUNS_WITH_NO_CHANGE_IN_LOSS
        last_epoch_mse_loss = None

        for epoch in range(self.max_epochs):
            for k in range(max(self.k_fold, 1)):
                train_loader, valid_loader = split_train_validation(x_train_df, y_train_df, tag_name,
                                                                    k_fold=self.k_fold)
                train_k_loss, Yhat_train, Y_gt_train = self._train(train_loader, train_step)
                if self.k_fold > 0:
                    valid_k_loss, last_epoch_mse_loss, corr, r2, Yhat_valid, Y_gt_valid = self._test(valid_loader,
                                                                                                     index_of_additional_parameters)

            test_loss, _, _, _, Yhat_test, Y_gt_test = self._test(test_loader, -1)

            test_losses.append(test_loss / len(test_loader.dataset[0][0]))
            train_losses.append((train_k_loss / max(self.k_fold, 1)) / len(train_loader.dataset[0][0]))
            if self.k_fold > 0:
                valid_losses.append((valid_k_loss / max(self.k_fold, 1)) / len(valid_loader.dataset[0][0]))

                # stopping rule-min does not decrease for 20 epochs:
                num_of_runs_with_no_change -= 1
                if min_loss > (valid_k_loss / max(self.k_fold, 1)) / len(valid_loader.dataset[0][0]):
                    min_loss = (valid_k_loss / max(self.k_fold, 1)) / len(valid_loader.dataset[0][0])
                    num_of_runs_with_no_change = NUM_OF_RUNS_WITH_NO_CHANGE_IN_LOSS
                print(num_of_runs_with_no_change)
                if num_of_runs_with_no_change <= 0:
                    break
        # ploting pred vs tag:
        scatter_pred_vs_tag(Yhat_train, Y_gt_train, title="train")
        scatter_pred_vs_tag(Yhat_valid, Y_gt_valid, title="validation")
        scatter_pred_vs_tag(Yhat_test, Y_gt_test, title="test")
        return train_losses, test_losses, valid_losses, (last_epoch_mse_loss / max(self.k_fold, 1)) / len(
            valid_loader.dataset[0][0]), corr, r2


def censoring_nn(censor_df: pd.DataFrame, uncensor_df: pd.DataFrame, tag_name: str, n_epochs, title, title_heatmap,
                 k_fold: int = 0):
    """
    1. hyper parameters tunning for :
    alpha, beta, lr, wd
    2. plot losses of train test and validation as a function of number of epochs
    3. print the mse of last epoch of validation
    4. create heat map
    :param censor_df: data frame of censored samples
    :param uncensor_df: data frame of uncensored samples
    :param tag_name: name of tag
    :param n_epochs: number of epochs for the train and tests
    :@param k_fold: int with the number of kfold needed if kfold is 0, there is no kfold
    :return:none

    Run Example:
    censoring_nn(saliva_censor_df, saliva_uncensor_df, "time_to_ttcgvhd", 100)
    """
    model = Model_luzon
    report = []
    arr_mse = list()
    arr_corr = list()
    arr_r2 = list()
    ###
    x_train_df, y_train_df, x_test_df, y_test_df, index_of_additional_parameters = split_train_test_censored_and_uncensord(
        censor_df, uncensor_df, tag_name, additional_cols_to_keep=["ttcgvhd_for_loss"])

    test_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(x_test_df.values).float(),
                                                   torch.from_numpy(y_test_df.values).float()),
                             batch_size=20)
    ###
    values = np.append(np.array(0), np.logspace(-10, 0, 11, base=2))
    # values_alpha= [0.01]
    # values_beta= [0.01]
    for alpha in range(len(values)):
        arr_mse.append([])
        arr_corr.append([])
        arr_r2.append([])
        for beta in range(len(values)):
            loss_fn = My_loss(values[alpha], values[beta])
            for lr in [0.001]:
                for wd in [0.1]:
                    nn_runner = custom_nn_runner(model, loss_fn, learning_rate=lr, weight_decay=wd, k_fold=k_fold,
                                                 epochs=n_epochs)
                    train_losses, test_losses, valid_losses, last_epoch_mse_loss, corr_valid, r2_valid = nn_runner.run_train(
                        x_train_df,
                        y_train_df,
                        index_of_additional_parameters,
                        test_loader,
                        tag_name)
                    report.append(f"lr: {lr}, wd: {wd}, mse_loss: {last_epoch_mse_loss}")
                    print(
                        f"lr: {lr}, wd: {wd}, mse_loss: {last_epoch_mse_loss}, alpha:{values[alpha]}, beta:{values[beta]}")
                    plot_and_save_losses(train_losses, test_losses, values[alpha], values[beta], lr, wd,
                                         valid_losses=valid_losses, title=title)
                    plot_and_save_losses(train_losses, test_losses, values[alpha], values[beta], lr, wd,
                                         valid_losses=valid_losses,
                                         plot_dots=True, title=title)
                    del nn_runner
                    arr_mse[alpha].append(last_epoch_mse_loss)
                    arr_corr[alpha].append(corr_valid)
                    arr_r2[alpha].append(r2_valid)
    [print(i) for i in report]
    print(arr_mse)
    create_heatmap(np.array(arr_mse), values, title="mse" + title_heatmap)
    create_heatmap(np.array(arr_corr), values, title="corr" + title_heatmap)
    create_heatmap(np.array(arr_r2), values, title="r2" + title_heatmap)

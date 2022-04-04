import random
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from torch import  nn
from sklearn.metrics import accuracy_score

def single_bacteria_custom_corr_for_missing_values(input, target):
    valid_input = [float(val) for i, val in enumerate(torch.squeeze(torch.tensor(input),0)) ]
    valid_target = [float(val) for i, val in enumerate(torch.squeeze(torch.tensor(target),0))]
    corr, p_value = spearmanr(valid_input, valid_target)
    return corr


def custom_rmse_for_missing_values(input, target):

    """
    #print(target)
    print(target.shape)
    print("loss input:")
    #print(input)
    print(input.shape)
    print("loss missing_values:")
    print(missing_values.shape)
    m = torch.mul(torch.sub(target, input), missing_values.float().reshape(torch.sub(target, input).shape))
    print("after mul of sub")
    #print(m)
    print(m.shape)
    n = torch.pow(m, 2)
    print("after pow")
    #print(n)
    print(n.shape)
    s = torch.pow(n, 0.5)
    print("after square")
    #print(s)
    print(s.shape)
    e = torch.mean(s)
    print("after mean")
    #print(e)
    print(e.shape)
    print("--------------------------------------------------------------")
    """
    loss = torch.mean(torch.pow(torch.sub(target, input), 2))
    return loss


def nn_custom_r2_for_missing_values(input, target):
    # loss = r2_score(target, input)
    # print(target.shape)
    # print(input.shape)

    # check if y shape has 2 dimensions - calculate R2 for each dim
    if len(input.shape) == 2:
        # print(r2_score(target.detach().numpy(), input.detach().numpy()))
        r2_list = []
        input = torch.t(input)
        target = torch.t(target)
        for input_feature, target_feature in zip(input, target):
            r2 = r2_score(target_feature.detach().numpy(), input_feature.detach().numpy())
            r2_list.append(r2)
        return r2_list

    else:
        r2 = r2_score(target.detach().numpy(), input.detach().numpy())
        """
        mean_of_the_observed_data = torch.mean(target)
        ss_tot = torch.sum(torch.pow(torch.sub(target, mean_of_the_observed_data), 2))
        # ss_reg = torch.sum(torch.pow(torch.sub((input, mean_of_the_observed_data), 2))
        ss_res = torch.sum(torch.pow(torch.sub(target, input), 2))
        r2 = torch.sub(1, torch.div(ss_res, ss_tot))
        """
        return r2


def multi_bacteria_custom_corr_for_missing_values(input, target):
    # average of the correlation at each time point:
    target_by_times = [[] for i in range(len(target[0]))]
    input_by_times = [[] for i in range(len(target[0]))]
    corr_by_time = []
    for s_i, (tar_sample, inp_sample) in enumerate(zip(target, input)):
        for t_i, (tar_time_point, inp_time_point) in enumerate(zip(tar_sample, inp_sample)):
            target_by_times[t_i].append(tar_time_point.detach().numpy())
            input_by_times[t_i].append(inp_time_point.detach().numpy())

    target_by_times = np.array(target_by_times)
    input_by_times = np.array(input_by_times)
    for tar_time_t, inp_time_t in zip(target_by_times, input_by_times):
        valid_input = [float(val) for i, val in enumerate(np.array(inp_time_t).flatten())]
        valid_target = [float(val) for i, val in enumerate(np.array(tar_time_t).flatten())]
        corr, p_value = spearmanr(valid_input, valid_target)
        corr_by_time.append(corr)

    # print(corr_by_time)
    # print(np.mean(corr_by_time))
    return np.nanmean(corr_by_time)

def single_bacteria_lstm_corr(input, target):
    flatten_input = []
    flatten_target = []
    for item in input:
        flatten_input += item
    for item in target:
        flatten_target += item
    return spearmanr(flatten_input, flatten_target)[0]
def multi_bacteria_lstm_corr(input, target):
    # average of the correlation at each time point:
    target_by_times = [[] for i in range(len(target))]
    input_by_times = [[] for i in range(len(target))]
    corr_by_time = []
    for s_i, (tar_sample, inp_sample) in enumerate(zip(target, input)):
        for t_i, (tar_time_point, inp_time_point) in enumerate(zip(tar_sample, inp_sample)):
            target_by_times[t_i].append(tar_time_point)
            input_by_times[t_i].append(inp_time_point)

    target_by_times = np.array(target_by_times)
    input_by_times = np.array(input_by_times)
    for bact_i in range(len(target_by_times[0][0])):
        valid_input = []
        valid_target = []
        for tar_time_t, inp_time_t in zip(target_by_times, input_by_times):
            if len(tar_time_t) >0:
                valid_input += [float(val) for i, val in enumerate(np.array(inp_time_t)[:,bact_i].reshape(-1).flatten())]
                valid_target += [float(val) for i, val in enumerate(np.array(tar_time_t)[:,bact_i].reshape(-1).flatten())]
        corr, p_value = spearmanr(valid_input, valid_target)
        corr_by_time.append(corr)

    # print(corr_by_time)
    # print(np.mean(corr_by_time))
    return np.nanmean(corr_by_time)

def cross_entropy(input, target):
    loss = nn.CrossEntropyLoss()
    return loss(input, target)

def top_1_accuracy(preds, labels):
    return accuracy_score(labels, preds)

if __name__ == "__main__":
    B = 16
    N = 5
    T = 100
    input = np.array([[[random.random() for i in range(T)] for j in range(N)] for q in range(B)])
    target = np.array([[[random.random() for i in range(T)] for j in range(N)] for q in range(B)])
    missing_values = np.array([[[1 for i in range(T)] for j in range(N)] for q in range(B)])

    multi_bacteria_custom_corr_for_missing_values(input, target, missing_values)
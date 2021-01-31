import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_heatmap(data, values, title, save_path=""):
    matplotlib.rc('xtick', labelsize=30)
    matplotlib.rc('ytick', labelsize=30)
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(data)

    ax.set_xticks(np.arange(len(values)))
    ax.set_yticks(np.arange(len(values)))
    ax.set_xticklabels(values)
    ax.set_yticklabels(values)
    plt.ylabel("alpha", fontsize=25)
    plt.xlabel("beta", fontsize=25)
    plt.title(title, fontsize=25)
    for i in range(len(data)):
        for j in range(len(data[0])):
            text = ax.text(j, i, round(data[i, j], 2), fontsize=25,
                           ha="center", va="center", color="w")
    fig.show()
    if save_path != "":
        os.makedirs(f"plots/{save_path}", exist_ok=True)
    fig.savefig(f"plots/{save_path}/" + title + ".svg", bbox_inches='tight')


def plot_and_save_losses(train_losses, test_losses, alpha, beta, lr, wd, title, valid_losses=None, plot_dots=False,
                         save_path="plots"):
    matplotlib.rc('xtick', labelsize=30)
    matplotlib.rc('ytick', labelsize=30)
    fig = plt.figure(figsize=(35, 10))
    if plot_dots:
        plt.plot(train_losses, "o", label="train")
        plt.plot(test_losses, "o", label="test")
        if valid_losses is not None:
            plt.plot(valid_losses, "o", label="validation")
    else:
        plt.plot(train_losses, label="train")
        plt.plot(test_losses, label="test")
        if valid_losses is not None:
            plt.plot(valid_losses, label="validation")
    plt.title(f"alpha: {alpha}, beta: {beta}, learning rate: {lr}, wight decay: {wd}", fontsize=30)
    plt.legend(fontsize=30)
    os.makedirs(f"{save_path}", exist_ok=True)
    plt.savefig(f"{save_path}/loss per epochs for alpha {alpha} beta {beta} lr {lr} wd {wd}" + title + ".svg")
    plt.show()
    fig.clear()
    plt.close(fig)


def scatter_pred_vs_tag(y_hat, y_gt, title, ax:plt.Axes=None, save_path="plots/new scatter pred tag"):
    """
    plot scatter of predict vs real tag
    @param y_hat: predict
    @param y_gt: real tag
    @param title: plot title
    @param save_path: where to save plot
    @return:
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.scatter(y_hat, y_gt)
    ax.set_title("predict vs real tag " + title)
    ax.set_xlabel("predict")
    ax.set_ylabel("real tag")
    ax.axis('square')
    if ax is None:
        fig.savefig(f"{save_path}/new scatter pred vs tag  " + title + ".svg")
        plt.show(fig)


def plot_histogram(id_col: pd.Series, time_vec, event_type_vec, parts, title, save_folder="final_plots/histograms"):
    df = pd.DataFrame([id_col, time_vec, event_type_vec]).T
    only_one_df = pd.DataFrame()
    for subject in df.groupby("subjid"):
        only_one_df = only_one_df.append(subject[1].iloc[0][1:])
    event_type_vec = only_one_df[only_one_df.columns[0]]
    time_vec = only_one_df[only_one_df.columns[1]]
    time_vec_0 = time_vec[event_type_vec == 0]
    time_vec_1 = time_vec[event_type_vec == 1]
    time_vec_2 = time_vec[event_type_vec == 2]
    time_vecs = [time_vec_0, time_vec_1, time_vec_2]
    if time_vec_0.max() < 10:
        time_vecs = [time_vec_0 * 365.25, time_vec_1 * 365.25, time_vec_2 * 365.25]

    dict_of_nums = {0: "Death", 1: title, 2: "Relapse"}
    range_ = (min(time_vecs[0].min(), time_vecs[1].min(), time_vecs[2].min()),
              max(time_vecs[0].max(), time_vecs[1].max(), time_vecs[2].max()))

    fig, axs = plt.subplots(2)
    axs[0].set_title("Distribution of times to " + dict_of_nums[1])
    axs[0].set_xlabel("time to event in days")
    axs[0].set_ylabel("frequency")
    axs[0].hist(time_vecs[1], parts, facecolor='green', alpha=0.5, ec="k", range=range_)

    axs[1].set_title("Distribution of times to competing event")
    axs[1].set_xlabel("time to event in days")
    axs[1].set_ylabel("frequency")

    axs[1].hist([time_vecs[0], time_vecs[2]], bins=parts, label=[dict_of_nums[0], dict_of_nums[2]], alpha=0.5,
                stacked=True, ec="k", range=range_)

    axs[1].legend()
    fig.tight_layout()
    plt.savefig(save_folder + "/" + title, bbox_inches='tight')
    plt.show()


def plot_fraction_of_censored_samples_after_competing_event_as_function_of_beta(beta_list, good_frac_bar, tag,
                                                                                save_folder="final_plots/data_augmentation_analysis"):
    os.makedirs(f"{save_folder}/{tag}", exist_ok=True)
    fig, ax1 = plt.subplots(1)
    ax1.plot(beta_list, good_frac_bar)
    ax1.set_title("fraction of censored sample after the competing event as a function of beta")
    ax1.set_xlabel("beta")
    ax1.set_ylabel("fraction of samples after competing event")
    plt.savefig(f"{save_folder}/{tag}" + "/fraction_of_censored_samples_after_competing_event_as_func_of_beta.svg",
                bbox_inches='tight')
    ax1.set_xscale('log')
    plt.show()


def plot_fraction_of_censored_samples_after_competing_event_as_a_function_of_lamda(lamda_list, good_frac, tag,
                                                                                   save_folder="final_plots/data_augmentation_analysis"):
    os.makedirs(f"{save_folder}/{tag}", exist_ok=True)
    fig, ax2 = plt.subplots(1)
    ax2.scatter(lamda_list, good_frac)
    ax2.set_title("fraction of censored sample after the competing event as a function of lambda")
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("fraction of samples after competing event")
    ax2.set_xscale('log', base=2)
    plt.savefig(f"{save_folder}/{tag}" + "/fraction_of_censored_samples_after_competing_event_as_func_of_lamda.svg",
                bbox_inches='tight')
    plt.show()

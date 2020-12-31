import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def create_heatmap(data,values,title):
    matplotlib.rc('xtick', labelsize=30)
    matplotlib.rc('ytick', labelsize=30)
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(data)

    ax.set_xticks(np.arange(len(values)))
    ax.set_yticks(np.arange(len(values)))
    ax.set_xticklabels(values)
    ax.set_yticklabels(values)
    plt.ylabel("alpha",fontsize=25)
    plt.xlabel("beta",fontsize=25)
    plt.title(title,fontsize= 25)
    for i in range(len(data)):
        for j in range(len(data[0])):
            text = ax.text(j, i, round(data[i, j],2), fontsize=25,
                           ha="center", va="center", color="w")
    fig.show()
    fig.savefig("plots/"+title+".svg",bbox_inches='tight')

def plot_and_save_losses(train_losses, test_losses, alpha, beta, lr, wd, title,valid_losses=None, plot_dots=False,
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
    plt.savefig(f"{save_path}/loss per epochs for alpha {alpha} beta {beta} lr {lr} wd {wd}"+title+".svg")
    plt.show()
    fig.clear()
    plt.close(fig)

def scatter_pred_vs_tag(y_hat,y_gt,title,save_path="plots/new scatter pred tag"):
    """
    plot scatter of predict vs real tag
    @param y_hat: predict
    @param y_gt: real tag
    @param title: plot title
    @param save_path: where to save plot
    @return:
    """
    plt.scatter(y_hat,y_gt)
    plt.title("predict vs real tag "+title)
    plt.xlabel("predict")
    plt.ylabel("real tag")
    plt.savefig(f"{save_path}/new scatter pred vs tag  "+title+".svg")
    plt.show()

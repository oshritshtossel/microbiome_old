from LearningMethods.CorrelationFramework import CorrelationFramework
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json

#send as a parameter a dictionary with treshold, positive(color as string), negative(color as string), real(color as a char), random(color as a char)
def use_corr_framwork(X: pd.DataFrame, y, dict={},title=None, folder=""):
    cf = CorrelationFramework(X, y)

    fig1 = plt.figure(figsize=(24, 18))
    plt.subplots_adjust(top=0.94, bottom=0.07, left=0.2, right=1.0, hspace=0.2, wspace=0.2)
    grid = plt.GridSpec(4, 8, wspace=0, hspace=1)
    ax1 = plt.subplot(grid[0:2, 1:3])
    ax2 = plt.subplot(grid[2:5, 0:3])
    ax3 = plt.subplot(grid[:5, 3:10])

    positive_dict = {'height': 0.8}
    negative_dict = {'height': 0.8}
    positive_dict['color'] = dict['positive'] if 'positive' in dict else 'green'
    negative_dict['color'] = dict['negative'] if 'negative' in dict else 'red'
    treshold = dict['treshold'] if 'treshold' in dict else 1
    real_color = dict['real'] if 'real' in dict else 'g'
    random_Color = dict['random'] if 'random' in dict else 'b'
    cf.plot.plot_positive_negative_bars(ax1, treshold, positive_dict=positive_dict, negative_dict=negative_dict)
    real_hist_dict = {'bins': 30, 'color': real_color, 'label': 'Real values', 'density': True, 'alpha': 0.5}
    shuffled_hist_dict = {'bins': 30, 'color': random_Color, 'label': 'Shuffled values', 'density': True, 'alpha': 0.5}
    cf.plot.plot_real_and_shuffled_hist(ax2, real_hist_dict=real_hist_dict, shuffled_hist_dict=shuffled_hist_dict)
    cf.plot.plot_graph(ax3, 0.5, tree_dict)

    if title is not None:
        fig1.suptitle(title, fontsize=16)
    if folder != '' and title is not  None:
        fig1.savefig(f"{folder}/{title}.svg")
    else:
        fig1.savefig("figure1.svg")

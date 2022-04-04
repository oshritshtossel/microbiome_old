import re
import pandas as pd
from matplotlib.lines import Line2D

from LearningMethods.correlation_evaluation import SignificantCorrelation
import Plot.plot_positive_negative_bars as PP
import Plot.plot_real_and_shuffled_hist as PPR
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt
from LearningMethods.New_taxtree_draw import draw_tree

class CorrelationFramework:
    def __init__(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        self.x = x.copy()
        self.y = y.copy()

        self.sc = SignificantCorrelation(self.x, self.y, **kwargs)
        self.correlation_tree = self.sc.get_real_correlations()
        self.plot = _CorrelationPlotter(self.sc, self.correlation_tree)


class _CorrelationPlotter:
    def __init__(self, significant_correlation, correlation_tree):
        self.significant_correlation = significant_correlation
        self.correlation_tree = correlation_tree

    def plot_positive_negative_bars(self, ax: Axes, tree_dict, last_taxonomic_levels_to_keep=2, **kwargs):
        significant_bacteria = self.significant_correlation.get_most_significant_coefficients(
        percentile=tree_dict["percentile"])
        u = 2
        while len(significant_bacteria) > 15 and u < 8:
            significant_bacteria = self.significant_correlation.get_most_significant_coefficients(
            percentile=min(tree_dict["percentile"] / u, 0.25))
            u *= 2
        if last_taxonomic_levels_to_keep is not None:
            significant_bacteria.index = [delete_empty_taxonomic_levels(str(i)) for i in significant_bacteria.index]
            significant_bacteria.index = [i + "  (" + level(i) + ")" for i in significant_bacteria.index]
            # significant_bacteria.index = [delete_suffix(str(i))
            # for i in significant_bacteria.index]
            new_ind = []
            for ind in range(len(significant_bacteria.index)):
                if re.search(r'[a-zA-Z][a-zA-Z]',','.join(str(significant_bacteria.index[ind]).split(';')[-last_taxonomic_levels_to_keep:]).replace(",",", ").replace("[", "").replace("]", "")) == None:
                    new_ind.append(','.join(str(significant_bacteria.index[ind]).split(';')[-last_taxonomic_levels_to_keep - 1:-last_taxonomic_levels_to_keep]).replace(",", ", ").replace("[", "").replace("]", "") + "  (" + level(significant_bacteria.index[ind]) + ")")
                else:
                    new_ind.append(significant_bacteria.index[ind])
            significant_bacteria.index = new_ind
            significant_bacteria.index = [str(','.join(str(i).split(';')[-last_taxonomic_levels_to_keep:])).replace(",", ", ").replace("[", "").replace("]","") for i in significant_bacteria.index]
            significant_bacteria.index = [re.sub(r'[a-zA-Z]_+', '', str(i)) for i in significant_bacteria.index]
            print(significant_bacteria.index)

            return PP.plot_positive_negative_bars(ax, significant_bacteria, **kwargs)

    def clean_correlation_framework(self):
        self.correlation_tree = self.correlation_tree[self.correlation_tree.index.str.match(r'^(([^;]+);)+[^;]+$')]

    def plot_graph(self, ax: Axes, threshold=0.0, dict={}):
        if not dict:
            dict = {"netural": "yellow", "positive": "blue", "negative": "red", "treshold": 1.0}
        if "netural" not in dict:
            dict["netural"] = "black"
        if "positive" not in dict:
            dict["positive"] = "blue"
        if "negative" not in dict:
            dict["negative"] = "red"
        if "colormap" not in dict:
            dict['colormap'] = 'YlOrRd'
        self.clean_correlation_framework()
        custom_lines = [Line2D([0], [0], color='blue', lw=4),
                         Line2D([0], [0], color='black', lw=4),
                         Line2D([0], [0], color='red', lw=4)]
        ax.legend(custom_lines, ['Positive', 'Neutral', 'Negative'], loc='lower right', prop={'size': 18})
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        return draw_tree(ax, self.correlation_tree, dict, apply_purne=True)


    def plot_real_and_shuffled_hist(self, ax: Axes, **kwargs):
            return PPR.plot_real_and_shuffled_hist(ax, self.significant_correlation.coeff_df['real'],
                                                   self.significant_correlation.coeff_df.drop('real',
                                                                                          axis=1).values.flatten(),
                                               **kwargs)

def delete_empty_taxonomic_levels(i):
    splited = i.split(';')
    while re.search(r'[a-z]_+\d*$', splited[-1]) is not None:
        splited = splited[:-1]
    i = ""
    for j in splited:
        i += j
        i += ';'
    i = i[:-1]
    return i

def delete_suffix(i):
    m = re.search(r'_+\d+$', i)
    if m is not None:
        i = i[:-(m.end()-m.start())]
    return i

def level(i):
    lst = i.split(';')
    last = lst[-1].replace(" ", "")
    return last[0]

def use_corr_framwork(X: pd.DataFrame, y, title=None, folder="plots"):
    cf = CorrelationFramework(X, y)

    fig1 = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    positive_dict = {'color': 'blue', 'height': 0.8}
    negative_dict = {'color': 'red', 'height': 0.8}
    cf.plot.plot_positive_negative_bars(ax1, 1, positive_dict=positive_dict, negative_dict=negative_dict)
    real_hist_dict = {'bins': 30, 'color': 'g', 'label': 'Real values', 'density': True, 'alpha': 0.5}
    shuffled_hist_dict = {'bins': 30, 'color': 'b', 'label': 'Shuffled values', 'density': True, 'alpha': 0.5}
    cf.plot.plot_real_and_shuffled_hist(ax2, real_hist_dict=real_hist_dict, shuffled_hist_dict=shuffled_hist_dict)

    if title is not None:
        fig1.suptitle(title.replace("_", " "), fontsize=16)

    fig1.tight_layout()
    fig1.savefig(f"{folder}/{title}.svg")

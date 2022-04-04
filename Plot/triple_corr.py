import matplotlib.pyplot as plt
import pandas as pd
import sys
import json
import CorrelationFramework
from IPython.display import set_matplotlib_formats

def use_corr_framwork(X: pd.DataFrame, y, dict={}, title=None, folder=""):
    cf = CorrelationFramework.CorrelationFramework(X, y)

    fig1 = plt.figure(figsize=(24, 18))
    grid = plt.GridSpec(4, 8, wspace=0, hspace=1)
    ax1 = plt.subplot(grid[0:2, 1:3])
    ax2 = plt.subplot(grid[2:5, 0:3])
    ax3 = plt.subplot(grid[:5, 3:10])
    fig1.subplots_adjust(top=0.94, bottom=0.07, left=0.2, right=1.0, hspace=0.2, wspace=0.2)

    positive_dict = {'height': 0.8}
    negative_dict = {'height': 0.8}
    positive_dict['color'] = dict['positive'] if 'positive' in dict else 'red'
    negative_dict['color'] = dict['negative'] if 'negative' in dict else 'black'

    treshold = dict['treshold'] if 'treshold' in dict else 1
    real_color = dict['real'] if 'real' in dict else 'g'
    random_Color = dict['random'] if 'random' in dict else 'b'
    tree_dict = {"treshold": 0.15, "percentile": 2}# was 0.3
    tree = cf.plot.plot_graph(ax3, treshold, tree_dict)
    # plt.show()
    if tree == None:
        fig1 = plt.figure(figsize=(24, 18))
        fig1.subplots_adjust(top=0.94, bottom=0.07, left=0.2, right=1.0, hspace=0.2, wspace=0.2)
        grid = plt.GridSpec(4, 3, wspace=0.5, hspace=1)
        ax1 = plt.subplot(grid[0:2, 1])
        ax2 = plt.subplot(grid[2:4, 0:2])
        # fig1.show()
    else :
        fig1.subplots_adjust(top=0.94, bottom=0.07, left=0.2, right=1.0, hspace=0.2, wspace=0.2)
    cf.plot.plot_positive_negative_bars(ax1, tree_dict,last_taxonomic_levels_to_keep=1, positive_dict=positive_dict, negative_dict=negative_dict)

    real_hist_dict = {'bins': 30, 'color': real_color, 'label': 'Real values', 'density': True, 'alpha': 0.5}
    shuffled_hist_dict = {'bins': 30, 'color': random_Color, 'label': 'Shuffled values', 'density': True, 'alpha': 0.5}
    cf.plot.plot_real_and_shuffled_hist(ax2, real_hist_dict=real_hist_dict, shuffled_hist_dict=shuffled_hist_dict)
    fig1.subplots_adjust(top=0.94, bottom=0.07, left=0.2, right=1.0, hspace=0.2, wspace=0.2)
    # fig1.show()
    if title is not None:
        fig1.suptitle(title.replace(' ', '_'), fontsize=12)
    if folder != '' and title is not None:
        fig1.savefig(f"{folder}/{title}.pdf", pdi=400)
    else:
        fig1.savefig("figure1.pdf", dpi=400)

def main(df, tag, name='', folder=''):
    tree_dict = {"treshold": 1.0, "percentile": 1.0}
    if len(sys.argv) >= 2:
        tree_dict = json.loads(sys.argv[1])

    # meta = pd.concat((pd.read_csv("../../swaps/gMic/split_datasets/Vaginal_swaps_split_dataset/merged_train_tags.csv", index_col=0), pd.read_csv("../../swaps/gMic/split_datasets/Vaginal_swaps_split_dataset/merged_test_tags.csv", index_col=0)))['Tag']
    # df = pd.concat((pd.read_csv("../../swaps/gMic/split_datasets/Vaginal_swaps_split_dataset/merged_train_microbiome.csv", index_col=0), pd.read_csv("../../swaps/gMic/split_datasets/Vaginal_swaps_split_dataset/merged_test_microbiome.csv", index_col=0))).iloc[:, :-6]
    # meta = meta.loc[df.index]
    # meta = pd.read_csv('../Plot/t.csv', index_col=0)['Tag']
    # df = pd.read_csv('../Plot/f.csv', index_col=0)
    # tag = meta

    # make the tag binary
    use_corr_framwork(df.loc[list(set(df.index).intersection(set(tag.index)))],
                      tag.loc[list(set(df.index).intersection(set(tag.index)))], tree_dict, title=name, folder=folder)

if __name__ == "__main__":
    df = pd.read_csv('../../saliva/data/saliva_full_data.csv')
    meta = pd.read_csv('../../saliva/data/israel_saliva_meta.csv')
    df.index = df['ID']
    df = df.iloc[:,1:]
    meta.index = meta['ID']
    df = df.reindex(meta.index)
    df = df.dropna(how='all')
    meta = meta.reindex(df.index)
    meta = meta.replace({'GDM':1, 'Control':0})


    israel_all = [df[meta['Country'] == 'Israel'], meta[meta['Country'] == 'Israel']['Control_GDM'], 'GDM Israel all']
    russia_all = [df[meta['Country'] == 'Israel'], meta[meta['Country'] == 'Israel']['Control_GDM'], 'GDM Russia all']
    israel_t3 = [df[(meta['Country'] == 'Israel') & (meta['trimester'] == 'T3')], meta[(meta['Country'] == 'Israel') & (meta['trimester'] == 'T3')]['Control_GDM'], 'GDM Israel T3']
    russia_t3 = [df[(meta['Country'] == 'Israel')& (meta['trimester'] == 'T3')], meta[(meta['Country'] == 'Israel')& (meta['trimester'] == 'T3')]['Control_GDM'], 'GDM Russia T3']
    israel_t2 = [df[(meta['Country'] == 'Israel') & (meta['trimester'] == 'T2')], meta[(meta['Country'] == 'Israel') & (meta['trimester'] == 'T2')]['Control_GDM'], 'GDM Israel T2']
    russia_t2 = [df[(meta['Country'] == 'Russia')& (meta['trimester'] == 'T2')], meta[(meta['Country'] == 'Russia')& (meta['trimester'] == 'T2')]['Control_GDM'], 'GDM Russia T2']

    meta = meta.replace({'Israel':0, 'Russia':1})
    israel_vs_russia_t3 = [df[meta['trimester'] == 'T3'], meta[meta['trimester'] == 'T3']['Country'], 'Israel vs Russia T3']

    meta = meta.replace({'T2':0, 'T3':1})
    t2_vs_t3_israel = [df[(meta['trimester'] != 'T1') & (meta['Country'] == 0)], meta[(meta['trimester'] != 'T1') & (meta['Country'] == 0)]['trimester'], 'T2 vs T3 Israel']
    t2_vs_t3_russia = [df[(meta['trimester'] != 'T1') & (meta['Country'] == 1)], meta[(meta['trimester'] != 'T1') & (meta['Country'] == 1)]['trimester'], 'T2 vs T3 Russia']

    to_run = [israel_all, russia_all, israel_t3, russia_t3, israel_t2, russia_t2, israel_vs_russia_t3, t2_vs_t3_israel, t2_vs_t3_russia, israel_t2, russia_t2]
    for run in to_run:
        main(run[0], run[1], run[2], folder='saliva')

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Preprocess.preprocess_grid import row_normalization, fill_taxonomy
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles

def plot_rel_freq(data_frame, taxonomy_col="taxonomy", tax_level=3, folder=None):
        taxonomy_reduced = data_frame[taxonomy_col].map(lambda x: x.split(';'))
        taxonomy_reduced = taxonomy_reduced.map(lambda x: ';'.join(x[:tax_level]))
        data_frame[taxonomy_col] = taxonomy_reduced
        data_frame = data_frame.groupby(data_frame[taxonomy_col]).mean()
        data_frame = data_frame.T
        data_frame = row_normalization(data_frame)
        plotting_with_pd(data_frame, folder, tax_level)


def plotting_with_pd(df: pd.DataFrame, folder=None, taxonomy_level=3):
    df = easy_otu_name(df)
    df = df.reindex(df.mean().sort_values().index, axis=1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})
    ax2.axis('off')
    df.plot.bar(stacked=True, ax=ax, width=1.0, colormap='Spectral')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="x-small")
    ax.xaxis.set_ticks([])
    ax.set_xlabel("")
    ax.set_title("Relative frequency with taxonomy level "+str(taxonomy_level))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    #plt.show()
    plt.savefig(f"{folder}/relative_frequency_stacked.png")

def easy_otu_name(df):
    df.columns = [str([h[4:].capitalize() for h in i.split(";")][-2:])
                    .replace("[", "").replace("]", "").replace("\'", "") for i in df.columns]
    return df


if __name__ == "__main__":
    with open(Path('otuMF_6'), 'rb') as otumf_file:
        otumf = pickle.load(otumf_file)
        data = otumf.otu_features_df
        as_data_frame = pd.DataFrame(data.T).apply(pd.to_numeric, errors='ignore').copy()  # data frame of OTUs
        as_data_frame = fill_taxonomy(as_data_frame, tax_col="taxonomy")
        plot_rel_freq(as_data_frame)
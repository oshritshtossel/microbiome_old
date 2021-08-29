import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import preprocess_grid


def split_according_to_feature(dataframe, file, splitter, splitter2=None, spaces=10):
    df = pd.read_csv(file)
    controls = df.loc[df[splitter] == "normal"]["ID"]
    others = df.loc[df[splitter] != "normal"]["ID"]
    df_control = dataframe.loc[dataframe.index.isin(controls.array)]
    df_other = dataframe.loc[dataframe.index.isin(others.array)]
    if splitter2:
        df_control = split_according_to_feature(df_control, file, splitter2)
        df_other = split_according_to_feature(df_other, file, splitter2)
    df_filler = pd.DataFrame(0, columns=dataframe.columns,index=np.arange(spaces))
    new_df = df_control.append(df_filler).append(df_other)
    return new_df




def plot_rel_freq(data_frame, taxonomy_col="taxonomy", tax_level=3, folder=None, split=None):
        taxonomy_reduced = data_frame[taxonomy_col].map(lambda x: x.split(';'))
        taxonomy_reduced = taxonomy_reduced.map(lambda x: ';'.join(x[:tax_level]))
        data_frame[taxonomy_col] = taxonomy_reduced
        data_frame = data_frame.groupby(data_frame[taxonomy_col]).mean()
        data_frame = data_frame.T
        data_frame = row_normalization(data_frame) #preprocess_grid.
        plotting_with_pd(data_frame, folder, tax_level, split)


def plotting_with_pd(df: pd.DataFrame, folder=None, taxonomy_level=3, split=None):
    df = easy_otu_name(df)
    df = df.reindex(df.mean().sort_values().index, axis=1)

    if split:
        if type(split) == str:
            df = split_according_to_feature(df, "mapping_table.csv", split)
        else:
            df = split_according_to_feature(df, "mapping_table.csv", split[0], split[1])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})
    ax2.axis('off')
    df.plot.bar(stacked=True, ax=ax, width=1.0, colormap='Spectral')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="x-small")
    ax.xaxis.set_ticks([])
    ax.set_xlabel("")
    ax.set_title("Relative frequency with taxonomy level " + str(taxonomy_level))

    if split:
        if type(split) == str:
            ax.set_xlabel(f"Control                                  CAC                     ")
            ax.set_title(f"Relative frequency with taxonomy level {str(taxonomy_level)} \nsplit"
                         f" according to {split}")
        else:
            ax.set_xlabel(f"Control; Control,   Control; CAC,   CAC; Control,   CAC; CAC")
            ax.set_title(f"Relative frequency with taxonomy level {str(taxonomy_level)} \nsplit"
                         f" according to {split[0]} and {split[1]}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    #plt.show()
    plt.savefig(f"{folder}/relative_frequency_stacked.png")


def easy_otu_name(df):
    df.columns = [str([h.lstrip("kpcofgs").lstrip("_").capitalize() for h in i.split(";")][-2:])
                    .replace("[", "").replace("]", "").replace("\'", "") for i in df.columns]
    return df

def row_normalization(as_data_frame):
    as_data_frame = as_data_frame.div(as_data_frame.sum(axis=1), axis=0).fillna(0)
    return as_data_frame


if __name__=="__main__":
    data_frame = pd.read_csv("otu.csv", index_col="ID")
    data_frame = pd.DataFrame(data_frame.T).apply(pd.to_numeric, errors='ignore').copy()
    plot_rel_freq(data_frame, split=("Treatment", "Transplant"))

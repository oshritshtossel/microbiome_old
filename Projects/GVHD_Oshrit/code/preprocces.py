from typing import Tuple, Dict

import pandas as pd
from dateutil import relativedelta

from LearningMethods import create_otu_and_mapping_files


def preprocces(otu_table: pd.DataFrame, mapping: pd.DataFrame, tag_name: str, task_name: str, folder: str,
               tax: int = 6, taxnomy_group: str = 'sub PCA', epsilon: float = 0.1, normalization: str = "log",
               z_scoring: str = "row", norm_after_rel: str = "No", std_to_delete: int = 0,
               pca: Tuple[int, str] = (0, 'PCA')) -> Tuple[pd.DataFrame, pd.DataFrame, any]:
    """
    This function is a convenient fassad for the preprocess in microbiome git.

    :param otu_table: data frame of otu table with an "ID" column for the samples,the name of
                      columns are indexes(numbers), the names of bacterias are the last row
                      The result of - preprocces_otu_table
    :param mapping: the target column of the tag is named:"Tag"
                    The result of - preprocces_mapping
    :param tag_name: name of tag
    :param task_name: a name for the saving
    :param folder: name of folder to save at
    :param tax: taxonomy level a int between 2 to 7
    :param taxnomy_group: default 'sub PCA'
    :param epsilon:
    :param normalization:
    :param z_scoring:
    :param norm_after_rel:
    :param std_to_delete:
    :param pca:
    :return: 2 or 3 data frames : otu df , tag df, pca df
    """
    otu_table = preprocces_otu_table(otu_table)
    mapping = preprocces_mapping(mapping, tag_name)

    otu_table.to_csv("preproccesed data/otu_table.csv")
    mapping.to_csv("preproccesed data/mapping.csv")

    data_creator = create_otu_and_mapping_files.CreateOtuAndMappingFiles("preproccesed data/otu_table.csv",
                                                                         "preproccesed data/mapping.csv")

    preprocess_params = {'taxonomy_level': tax, 'taxnomy_group': taxnomy_group, 'epsilon': epsilon,
                         'normalization': normalization, 'z_scoring': z_scoring, 'norm_after_rel': norm_after_rel,
                         'std_to_delete': std_to_delete, 'pca': pca}

    data_creator.preprocess(preprocess_params, False, taxonomy_col="ID")
    otu_path, tag_path, pca_path = data_creator.csv_to_learn(task_name, folder, tax)

    otu = pd.read_csv(otu_path, index_col=0)
    tag = pd.read_csv(tag_path, index_col=0)
    if pca_path != 'No pca created':
        pca = pd.read_csv(pca_path, index_col=0)
        return otu, tag, pca
    return otu, tag, None


def preprocces_otu_table(otu_table: pd.DataFrame) -> pd.DataFrame:
    """
    The func prepare the raw data frame of otu table to the proper form for preprocces function.
    It transpose the df, so that the name of bacterias will be the columns and the samples the raws.
    It changes the name of column of the samples to ID.
    It puts the names of the columns, the bacterias names to the last raw and put numbers as the names of columns
    instead
    :param otu_table: dataframe of otu table
    :return: otu table data frame in the correct shape for preprocess
    """
    otu_table = otu_table.T.reset_index()
    otu_table = otu_table.append(otu_table.iloc[0])
    otu_table = otu_table.rename(columns={"index": "ID"}, inplace=False)
    otu_table = otu_table.iloc[1:]
    return otu_table


def preprocces_mapping(mapping: pd.DataFrame, tag_name: str) -> pd.DataFrame:
    """
    The func prepare the raw data frame of mapping file to the proper form for preprocces function.
    :param mapping: data frame of mapping file
    :param tag_name: name of the tag column
    :return: mapping file data frame in the correct shape for preprocess
    """
    mapping = mapping.rename(columns={tag_name: "Tag"}, inplace=False)
    return mapping


def create_event_date_column(df: pd.DataFrame, event_column: str):
    """
    creates a column with the date of event , calculated by multiply the event column in years
    by 365.25 and calculating the date
    :param df: data frame we want to add a column to
    :param event_column: name of column of the times in years of the event
    :return: data frame with the column with the date of the event

    Run Example:
    stool_otu = create_event_date_column(stool_otu, "ttcgvhd")
    saliva_otu = create_event_date_column(saliva_otu, "ttcgvhd")
    """
    df["date_of" + event_column] = pd.to_datetime(df["bmtdate"]) + df[event_column].apply(
        lambda x: relativedelta.relativedelta(days=x * 365.25))
    return df


def classify_censoring(otu: pd.DataFrame, decision_column: str,
                       dict_of_censored: Dict[int, bool] = {0: True, 1: False,
                                                            2: True}) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The func returns a dataframe of only the censored samples and a dataframe of only the uncensored samples
    :param dict_of_censored: dict of int (the value in decision_column) to censored (True) or uncensored (False)
    :param otu: original otu df
    :return: 2 data frames: one of censored samples only and the other of uncensored samples
    """
    censored = otu.loc[otu[decision_column].map(dict_of_censored)]
    uncensored = otu.loc[~otu[decision_column].map(dict_of_censored)]

    return censored, uncensored



def data_augment_preprocess(otu: pd.DataFrame, tag: str, decision_column: str,
                            dict_of_censored: Dict[int, bool] = {0: True, 1: False,
                                                                 2: True}) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    the func prepare the data to the augment process by:
    1. getting rid of the nan in the tag column
    2. creating an event day column
    3. dividing the data to censored and uncensored
    :param otu: otu df
    :param tag: name of tag column
    :param decision_column: name of column with information about censored samples
    :param dict_of_censored: dictionary [int: censored/ uncensored]
    :return: 2 dfs ready to augment: censored df and uncensored df
    """
    # Remove null subjects according to column tag
    otu = otu.loc[otu[tag].fillna(-1) != -1]
    otu = create_event_date_column(otu, tag)
    return classify_censoring(otu, decision_column, dict_of_censored=dict_of_censored)


def delete_features_according_to_key(df: pd.DataFrame, keep=None, bacteria_col_keyword="k__Bacteria"):
    """
    delete all features which do not have the key word in their name
    :param df: dataframe
    :param bacteria_col_keyword: ""k__Bacteria"
    :param keep: a list of column names to keep
    :return: df with only the wanted columns
    """
    if keep:
        return df[
            [col for col in df.columns if bacteria_col_keyword in col or col in keep]]
    else:
        return df[
            [col for col in df.columns if bacteria_col_keyword in col]]


def add_tag_to_predict(censor_df, uncensor_df, tag, data_augmented=True):
    """
    add a tag of the time passes from the sample time to the event in uncensored:
    and in censored: the time passes from the sample time to the augmented event and a column for the time passes
    from the sample to competing event.
    :param censor_df: data frame of censor
    :param uncensor_df: data frame of uncensored
    :param tag: name of tag
    :return: censor_df, uncensor_df
    """
    # add column tag to censored
    if data_augmented:
        censor_df["time_to_" + tag] = ((pd.to_datetime(censor_df["bmtdate"]) + (censor_df["new_" + tag] * 365.25).apply(
            lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(censor_df["DATE"])) / pd.to_timedelta(1,
                                                                                                                   unit='D')
    else:
        censor_df["time_to_" + tag] = ((pd.to_datetime(censor_df["bmtdate"]) + (censor_df[tag] * 365.25).apply(
            lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(censor_df["DATE"])) / pd.to_timedelta(1,
                                                                                                                   unit='D')
    # add column  for loss to censored
    censor_df[tag + "_for_loss"] = ((pd.to_datetime(censor_df["bmtdate"]) + (censor_df[tag] * 365.25).apply(
        lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(censor_df["DATE"])) / pd.to_timedelta(1,
                                                                                                               unit='D')
    # add column to tag to uncensored
    uncensor_df["time_to_" + tag] = ((pd.to_datetime(uncensor_df["bmtdate"]) + (uncensor_df[tag] * 365.25).apply(
        lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(uncensor_df["DATE"])) / pd.to_timedelta(1,
                                                                                                                 unit='D')
    return censor_df, uncensor_df

# censor_df["time_to_" + tag] = (pd.to_datetime(censor_df["bmtdate"]) + (censor_df["new_" + tag] * 365.25).apply(
#      lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(censor_df["DATE"])
# uncensor_df["time_to_" + tag] = (pd.to_datetime(uncensor_df["bmtdate"]) + (
#       uncensor_df[tag] * 365.25).apply(
#   lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(uncensor_df["DATE"])

from typing import Dict, Tuple, Any
import pandas as pd

from Projects.GVHD_Oshrit.code import lerning_tool, preprocces


def load_and_preprocces_saliva_and_stool(saliva_otu_path, saliva_map_path, stool_otu_path, stool_map_path,
                                         word_from_column_to_delete: Dict[str, str] = None):
    """
    This func gets the pathes of our data otu tables and mapping files of microbiom from saliva and stool
    and returns us 6 data frames in a proper way to work with.

    :param saliva_otu_path: path of saliva otu
    :param saliva_map_path: path of saliva mapping file
    :param stool_otu_path: path of stool otu
    :param stool_map_path: path of stool mapping file
    :param word_from_column_to_delete: a dictionary from the form of column:word to delete from. It can delete
           only 1 word from each column per time
    :return: 6 ready to work with data frames

    Example of usage:
    main_functions.load_and_preprocces_saliva_and_stool("raw data/Saliva_OTU_table_220620.txt",
                                                        "raw data/saliva_mapping file.tsv",
                                                        "raw data/stool_OTU_table.tsv",
                                                        "raw data/stool_mapping_file.tsv", {"DonorType": "Haplo"})
    """
    saliva_OTU_table = pd.read_csv(saliva_otu_path, sep="\t")
    saliva_mapping = pd.read_csv(saliva_map_path, sep="\t")
    stool_OTU_table = pd.read_csv(stool_otu_path, sep="\t")
    stool_mapping = pd.read_csv(stool_map_path, sep="\t")
    if stool_mapping.columns[0] == "#SampleID":
        stool_mapping = stool_mapping.rename(columns={"#SampleID": "ID"})

    # Delete samples with specific words from specific columns from df:
    if word_from_column_to_delete is not None:
        for key in word_from_column_to_delete.keys():
            saliva_mapping = saliva_mapping[
                ~saliva_mapping[key].str.contains(word_from_column_to_delete[key], na=False)]
            stool_mapping = stool_mapping[~stool_mapping[key].str.contains(word_from_column_to_delete[key], na=False)]

    saliva_otu, saliva_tag, saliva_pca = preprocces.preprocces(saliva_OTU_table, saliva_mapping, "ttext_cgvhd",
                                                               "saliva", "data", tax=5, taxnomy_group="mean",
                                                               z_scoring='No')
    stool_otu, stool_tag, stool_pca = preprocces.preprocces(stool_OTU_table, stool_mapping, "ttext_cgvhd", "stool",
                                                            "data", tax=5, taxnomy_group="mean", z_scoring='No',epsilon=1)
    return saliva_otu, saliva_tag, saliva_pca, stool_otu, stool_tag, stool_pca


def preprocces_and_augment(otu: pd.DataFrame, tag: str, decision_column: str, augmentor: lerning_tool.Augment,
                           beta, lamda=None, dict_of_censored: Dict[int, bool] = {0: True, 1: False, 2: True}) -> Tuple[
    Any, Any, Any]:
    """
    the func prepares the data to the augment process by:
    # preprocess:
    1. getting rid of the nan in the tag column
    2. creating an event day column
    3. dividing the data to censored and uncensored
    #create an augment:
    create artificial time for the censored samples.
    # Implement the augment:
    adds the new column of time in years to event to  censored data frame.
    # Add tag to predict
    adds a column of our new tag to df, while the new tag is the time in days from the sample day
    to get the event. It is done both to the censored and uncensored dfs.
    The func returns to ready dfs: censored and uncensored.
    :param otu: saliva otu df or stool otu df that we load from data directory. These dfs are after load and first preprocess
    that is done by "load_and_preprocces_saliva_and_stool"
    :param tag: name of tag
    :param decision_column: the column which gives us information about 0-dead 1-event 2-competing event
    :param augmentor:
    :param beta: hyper parameter for the formula of augment data
    :param dict_of_censored: for example:{0:True (censored),1:False (uncensored),2:True (censored)}
    :return: ready to learn censored df and uncensored df

    Example Run:
    saliva_censor_df, saliva_uncensor_df = main_functions.preprocces_and_augment(saliva_otu, "ttcgvhd", "c_cgvhd",
                                                                                 lerning_tool.Bar_Augment, 0.5)
    stool_censor_df, stool_uncensor_df = main_functions.preprocces_and_augment(stool_otu, "ttcgvhd", "c_cgvhd",
                                                                               lerning_tool.Bar_Augment, 0.5)
    """
    # Preprocess
    censor_df, uncensor_df = preprocces.data_augment_preprocess(otu, tag, decision_column, dict_of_censored)

    # Crate an augment
    augmentor_instance = augmentor(otu, tag, censor_df, uncensor_df, beta)

    # Implement the augment
    try:
        censor_df = augmentor_instance.implement_augment(censor_df, lamda)
    except:
        censor_df = augmentor_instance.implement_augment(censor_df)
    # Add tag to predict
    censor_df, uncensor_df = augmentor_instance.add_tag_to_predict(censor_df, uncensor_df)

    return censor_df, uncensor_df, augmentor_instance


def count_events_competing_events(df, column_name) -> dict:
    d_count = {0: 0, 1: 0, 2: 0}
    for subject in df.groupby(df["subjid"]):
        compete_val = subject[1].iloc[0][column_name]
        d_count[compete_val] += 1
    return d_count

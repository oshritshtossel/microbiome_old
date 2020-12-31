import os

import pandas as pd
import numpy as np

from Projects.GVHD_Oshrit.code import main_functions, lerning_tool
from Projects.GVHD_Oshrit.code.main_functions import load_and_preprocces_saliva_and_stool
from Projects.GVHD_Oshrit.code.neaural_network import censoring_nn
from Projects.GVHD_Oshrit.code.plotting_tools import create_heatmap
from Projects.GVHD_Oshrit.code.preprocces import preprocces_otu_table, preprocces_mapping, preprocces
from Projects.GVHD_Oshrit.code.utils import in_debug

if __name__ == '__main__':

    # Load
    # saliva_otu = pd.read_csv("data/OTU_merged_saliva.csv", index_col=0)
    stool_otu = pd.read_csv("data/NEW_OTU_merged_stool.csv", index_col=0)

    # Preprocess
    # saliva_censor_df, saliva_uncensor_df = main_functions.preprocces_and_augment(saliva_otu, "ttcgvhd", "c_cgvhd",
    #                                                                              lerning_tool.Bar_Augment, 0.5)
    for tag,time_to_tag in zip(["ttcgvhd"],["c_cgvhd"]):
        for beta in [0.1]:
            print(f'beta={beta}:')
            stool_censor_df, stool_uncensor_df, augmentor = main_functions.preprocces_and_augment(stool_otu, tag,
                                                                                                  time_to_tag,
                                                                                                  lerning_tool.MLE_Augment,
                                                                                                  beta,lamda=0.01)
            x=5

            # censoring_nn(saliva_censor_df, saliva_uncensor_df, "time_to_ttcgvhd", 100)
            censoring_nn(stool_censor_df, stool_uncensor_df, "time_to_ttcgvhd", 1000, k_fold=3,
                         title="mle without censor in days sigma 2mean lamda 0.001 ",
                         title_heatmap="heatmap mle without censor in days sigma 2mean lamda 0.001 ")

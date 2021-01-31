import os

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from Projects.GVHD_Oshrit.code import main_functions, lerning_tool
from Projects.GVHD_Oshrit.code.cemetry import try_svr
from Projects.GVHD_Oshrit.code.main_functions import load_and_preprocces_saliva_and_stool
from Projects.GVHD_Oshrit.code.neaural_network import censoring_nn, censoring_nn_leave_one_out, split_by_id
from Projects.GVHD_Oshrit.code.plotting_tools import create_heatmap, plot_histogram, \
    plot_fraction_of_censored_samples_after_competing_event_as_function_of_beta, \
    plot_fraction_of_censored_samples_after_competing_event_as_a_function_of_lamda
from Projects.GVHD_Oshrit.code.preprocces import preprocces_otu_table, preprocces_mapping, preprocces
from Projects.GVHD_Oshrit.code.utils import in_debug
import Projects.GVHD_Oshrit.code.main_functions

if __name__ == '__main__':
    # Load
    # saliva_otu, saliva_tag, saliva_pca, stool_otu, stool_tag, stool_pca = load_and_preprocces_saliva_and_stool("raw data/Saliva_OTU_table_220620.txt","raw data/saliva_mapping file.tsv","raw data/stool_OTU_table.tsv","raw data/stool_mapping_file_Adi.tsv")
    # stool_otu.to_csv("data/New_OTU_merged_stool.csv", index_col=0)

    org_stool_otu = pd.read_csv("data/New_OTU_merged_stool.csv", index_col=0)
    # Preprocess
    # saliva_censor_df, saliva_uncensor_df = main_functions.preprocces_and_augment(saliva_otu, "ttcgvhd", "c_cgvhd",
    #                                                                              lerning_tool.Bar_Augment, 0.5)
    for tag, time_to_tag, in zip(
            ["ttcgvhd", "tt_gutagvhd2c", "tt_gutagvhd3c", "ttoral_cgvhd", "ttext_cgvhd", "agvhd1_IBMTR_IId",
             "agvhd1_IBMTR_IIId", "agvhd1_GLUCKSBERG_IId", "agvhd1_GLUCKSBERG_IIId"]
            , ["c_cgvhd", "gutagvhd2c", "gutagvhd3c", "c_oral_cgvhd", "c_ext_cgvhd", "agvhd1_IBMTR_IIc",
               "agvhd1_IBMTR_IIIc",
               "agvhd1_GLUCKSBERG_IIc", "agvhd1_GLUCKSBERG_IIIc"]):
        stool_otu = org_stool_otu.loc[org_stool_otu[time_to_tag] != 0]
        # although it seems wrong it is good.tag="ttcgvhd" time_to_tag= "c_cgvhd"
        print("Working on " + tag)
        good_fracs = []
        good_fracs_bar = []
        # plot_histogram(stool_otu["subjid"], stool_otu[tag], stool_otu[time_to_tag], 30, tag)
        beta_list = [0.1]  # , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        lamda_list = [0.01]
        for beta in beta_list:
            for lamda in lamda_list:
                print(f'beta={beta}. lambda:{lamda}:')
                stool_censor_df, stool_uncensor_df, augmentor = main_functions.preprocces_and_augment(stool_otu, tag,
                                                                                                      time_to_tag,
                                                                                                      lerning_tool.MLE_Augment,
                                                                                                      beta, lamda)
                good_fracs.append(augmentor.good_frac)
                good_fracs_bar.append(augmentor.bar_good_frac)
                dict_compete = Projects.GVHD_Oshrit.code.main_functions.count_events_competing_events(stool_censor_df,
                                                                                                      time_to_tag)
                if beta == beta_list[0]:
                    dict_event = Projects.GVHD_Oshrit.code.main_functions.count_events_competing_events(
                        stool_uncensor_df,
                        time_to_tag)
                    print("Dict compete:")
                    print(dict_compete)
                    print("Dict event:")
                    print(dict_event)
                stool_uncensor_df.to_csv("data/stool_uncensored_df.csv")
                stool_censor_df.to_csv("data/stool_censored_df.csv")
                x=5
                # stool_censor_df = stool_censor_df.iloc[0:0]

                # censoring_nn_leave_one_out(stool_censor_df, stool_uncensor_df, "time_to_" + tag, 200)
                #y= pd.DataFrame()
                #y["subjid"] = stool_uncensor_df["subjid"]
                #y["Tag"]=stool_uncensor_df["time_to_ttcgvhd"]
                #y.to_csv("data/ttcgvhd_51_people.csv")
                #x=5
                try_svr(stool_uncensor_df, tag)
                exit(0)
                censoring_nn(stool_censor_df, stool_uncensor_df, "time_to_" + tag, 1000, k_fold=3,
                             title="mle without censor in days sigma 2mean lamda 0.001 ",
                             title_heatmap="heatmap mle without censor in days sigma 2mean lamda 0.001 ")
            # plot_fraction_of_censored_samples_after_competing_event_as_a_function_of_lamda(lamda_list, good_fracs, tag)
        # plot_fraction_of_censored_samples_after_competing_event_as_function_of_beta(beta_list, good_fracs_bar, tag)

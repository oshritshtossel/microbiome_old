import sys, os
import pandas as pd

from create_otu_and_mapping_files import CreateOtuAndMappingFiles
from diversity import Diversity
from preprocess_grid import update_state
preprocess_prms = {'taxonomy_level': 6, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1,
                   'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': 'No',
                   'std_to_delete': 0, 'pca': (0, 'PCA')}
'''
taxonomy_level 4-7 , taxnomy_group : sub PCA, mean, sum , epsilon: 0-1 
z_scoring: row, col, both , 'pca': (0, 'PCA') second element always PCA. first is 0/1 
normalization: log, relative , norm_after_rel: No, relative
'''


def evaluate(params, tag_flag, ip, only_tag):
    # updating the state - Creating otu And Mapping Files
    update_state(ip, 1)

    if tag_flag:
        mapping_file = CreateOtuAndMappingFiles(ip + "/OTU.csv", tags_file_path=None)
    else:
        mapping_file = CreateOtuAndMappingFiles(ip +"/OTU.csv", ip +"/TAG.csv")
    only_tag = not  only_tag
    mapping_file.preprocess(preprocess_params=params, visualize=only_tag, ip=ip)

    # updating the state - calculating diversities
    update_state(ip, 5)
    try:
        diversity(params["alpha_div"], params["beta_div"], ip)
    except:
        print("error occured while calculating the diversities")

    if not tag_flag:
        otu_path, mapping_path, pca_path = mapping_file.csv_to_learn("General_task", os.path.join(os.getcwd(),
                                                                                                 ip + "/General_files"),
                                                                     preprocess_prms['taxonomy_level'])
    else:
        otu_path, pca_path = mapping_file.csv_to_learn("General_task", os.path.join(os.getcwd(),ip + "/General_files"),
                                                       preprocess_prms['taxonomy_level'])
    print('CSV files are ready after MIPMLP')
    print('OTU file', otu_path)
    if not tag_flag:
        print('mapping file', mapping_path)
    print('PCA object file', pca_path)
    with open("./" + str(ip) + "/state.txt", "w+") as f:
        f.truncate(0)

def diversity(alpha, beta, ip):
    diversity = Diversity(ip +"/OTU.csv")
    diversity.plot_beta(beta, folder=str(ip) +"/static/").to_csv(ip +"/General_files/beta_div.csv")
    diversity.compute_alpha(alpha).to_csv(ip +"/General_files/alpha_div.csv")
    diversity.compute_beta(beta).to_csv(ip +"/General_files/beta_div.csv")


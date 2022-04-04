import sys, os
from create_otu_and_mapping_files import CreateOtuAndMappingFiles

direc = "General_files"
# These datasets were taken from Oshrit mail
datasets = ["cirrhosis"]
# datasets = ["bw"]

# for dataset in datasets:
#     os.mkdir(dataset)
for dataset in datasets:
    file_path = os.path.join(direc, f"{dataset}_for_preprocess.csv")
    # Log normalization
    preprocess_prms1 = {'taxonomy_level': 5, 'taxnomy_group': 'mean', 'epsilon': 1,
                         'normalization': 'log', 'z_scoring': 'No', 'norm_after_rel': 'No',
                         'std_to_delete': 0, 'pca': (0, 'PCA'), "rare_bacteria_threshold":-1}
    # Relative normalization
    preprocess_prms2 = {'taxonomy_level': 5, 'taxnomy_group': 'mean', 'epsilon': 1,
                         'normalization': 'relative', 'z_scoring': 'No', 'norm_after_rel': 'No',
                         'std_to_delete': 0, 'pca': (0, 'PCA'), "rare_bacteria_threshold":-1}

    otu_file = file_path
    tag_file = ''
    task_name = dataset

    mapping_file1 = CreateOtuAndMappingFiles(otu_file, tags_file_path=None)
    mapping_file1.preprocess(preprocess_params=preprocess_prms1, visualize=False, ip='127.0.0.1')
    mapping_file2 = CreateOtuAndMappingFiles(otu_file, tags_file_path=None)
    mapping_file2.preprocess(preprocess_params=preprocess_prms2, visualize=False, ip='127.0.0.1')

    sub = []
    mean = []
    for i in mapping_file1.otu_features_df_b_pca.columns:
        sub.append(i[:-2])
    for i in mapping_file2.otu_features_df_b_pca.columns:
        i = i.split(';')
        while len(i)>0 and i[-1][-1]=="_":
            i  = i[:-1]
        i = ';'.join(i)
        if i != '':
            mean.append(i)
    sub = set(sub)
    mean = set(mean)
    print(len(sub))
    print(len(mean))


    task1_name = f"{dataset}_after_mipmlp_taxonomy_{preprocess_prms1['taxonomy_level']}" \
                 f"group_{preprocess_prms1['taxnomy_group']}_epsilon_{preprocess_prms1['epsilon']}_" \
                 f"normalizaion_{preprocess_prms1['normalization']}"
    task2_name = f"{dataset}_after_mipmlp_taxonomy_{preprocess_prms2['taxonomy_level']}" \
                 f"group_{preprocess_prms2['taxnomy_group']}_epsilon_{preprocess_prms2['epsilon']}_" \
                 f"normalizaion_{preprocess_prms2['normalization']}"

    otu_path1, pca_path1 = mapping_file1.csv_to_learn(task1_name, os.path.join(os.getcwd(), task_name),
                                                                 preprocess_prms1['taxonomy_level'])

    otu_path2, pca_path2 = mapping_file2.csv_to_learn(task2_name, os.path.join(os.getcwd(), task_name),
                                                                  preprocess_prms2['taxonomy_level'])
    # print('CSV files are ready after MIPMLP')
    # print('OTU file', otu_path1)
    # print('PCA object file', pca_path1)
    #
    # print('OTU file', otu_path2)
    # print('PCA object file', pca_path2)
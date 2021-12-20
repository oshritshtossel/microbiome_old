import sys, os
from create_otu_and_mapping_files import CreateOtuAndMappingFiles
from  diversity import Diversity
import service
preprocess_prms = {'taxonomy_level': 6, 'taxnomy_group': 'mean', 'epsilon': 0.1, 'tax_level_plot': 4,
                     'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': 'No',
                     'std_to_delete': 0, 'pca': (0, 'PCA')}

otu_file = 'General_files/IBD_otus_for_preprocess.csv'
tag_file = 'General_files/tag_IBD_VS_ALL.csv'
task_name = 'General_task'

# mean = open('mean', 'r')
# sub = open('sub_pca', 'r')
# mean_lst = mean.read().split('\n')
# sub_lst = sub.read().split('\n')
# for l in range(len(sub_lst)):
#     sub_lst[l] = sub_lst[l][:-1]
# for l in mean_lst:
#     flag = 0
#     for j in sub_lst:
#        if l in j:
#             flag = 1
#             break
#     if not flag:
#         print(l)

mapping_file = CreateOtuAndMappingFiles(otu_file, tag_file)
mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=True, ip='127.0.0.1')
service.diversity("shannon", "unwieghted_unifrac", "127.0.0.1")


otu_path, mapping_path, pca_path = mapping_file.csv_to_learn(task_name, os.path.join(os.getcwd(), task_name), preprocess_prms['taxonomy_level'])
print('CSV files are ready after MIPMLP')
print('OTU file', otu_path)
print('mapping file', mapping_path)
print('PCA object file', pca_path)

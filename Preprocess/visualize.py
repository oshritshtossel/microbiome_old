import  matplotlib.pyplot as plt
import  numpy as np
import os
import seaborn as sns


taxonomy_col='taxonomy'
sub_plot_idx = [321, 322]
sub_plot_idx_after_tax_grouping = [323, 324]

class visualize:
    def __init__(self, as_data_frame, folder):
        self.folder = folder
        plt.figure('Preprocess')  # make preprocessing figure
        self.data_frame_for_vis = as_data_frame.copy()
        try:
            self.data_frame_for_vis = self.data_frame_for_vis.drop(taxonomy_col, axis=1)
        except:
            pass
        self.data_frame_flatten = self.data_frame_for_vis.values.flatten()
        self.visualize_preproccess('Before Taxonomy group', sub_plot_idx)

        self.data_frame_for_vis = as_data_frame.copy()

    def plot_after_taxonomy_grouping(self, as_data_frame):
        self.data_frame_flatten = as_data_frame.values.flatten()
        self.visualize_preproccess('After-Taxonomy - Before', sub_plot_idx_after_tax_grouping)
        self.visualize_preproccess('After-Taxonomy - Before', sub_plot_idx_after_tax_grouping)
        samples_density = as_data_frame.apply(np.sum, axis=1)
        plt.figure('Density of samples')
        samples_density.hist(bins=100, facecolor='Blue')
        plt.title(f'Density of samples')
        plt.savefig(os.path.join(self.folder, "density_of_samples.svg"), bbox_inches='tight', format="svg")
        plt.clf()

    def visualize_preproccess(self, name, subplot_idx):
        indexes_of_non_zeros = self.data_frame_flatten != 0
        plt.subplot(subplot_idx[0])
        self.plot_preprocess_stage(self.data_frame_flatten, name)
        result = self.data_frame_flatten[indexes_of_non_zeros]
        plt.subplot(subplot_idx[1])
        self.plot_preprocess_stage(result, name + ' without zeros')
        plt.clf()

    def plot_preprocess_stage(self, result, name, write_title=False, write_axis=True):
        plt.hist(result, 1000, facecolor='Blue', alpha=0.75)
        if write_title:
            plt.title('Distribution ' + name + ' preprocess')
        if write_axis:
            plt.xlabel('BINS')
            plt.ylabel('Count')

    def plot_var_histogram(self, as_data_frame):
        self.as_data_frame = as_data_frame.copy()
        samples_variance = self.as_data_frame.apply(np.var, axis=1)
        plt.figure('Variance of samples')
        samples_variance.hist(bins=100, facecolor='Blue')
        plt.title(
            f'Histogram of samples variance before z-scoring\nmean={samples_variance.values.mean()},'
            f' std={samples_variance.values.std()}')
        plt.savefig(os.path.join(self.folder, "samples_variance.svg"), bbox_inches='tight', format='svg')
        plt.clf()

    def plot_heatmaps(self, as_data_frame, taxonomy_level):
        self.as_data_frame = as_data_frame.copy()

        plt.figure('standard heatmap')
        sns.heatmap(as_data_frame, cmap="Blues", xticklabels=False, yticklabels=False)
        plt.title('Heatmap after standardization and taxonomy group level ' + str(taxonomy_level))
        plt.savefig(os.path.join(self.folder, "standard_heatmap.png"))
        plt.clf()
        corr_method = 'pearson'
        corr_name = 'Pearson'
        # if samples on both axis needed, specify the vmin, vmax and mathod
        plt.figure('correlation heatmap patient')
        sns.heatmap(as_data_frame.T.corr(method=corr_method), cmap='Blues', vmin=-1, vmax=1, xticklabels=False,
                    yticklabels=False)
        plt.title(corr_name + ' correlation patient with taxonomy level ' + str(taxonomy_level))
        # plt.savefig(os.path.join(folder, "correlation_heatmap_patient.svg"), bbox_inches='tight', format='svg')
        plt.savefig(os.path.join(self.folder, "correlation_heatmap_patient.png"))
        plt.clf()

        plt.figure('correlation heatmap bacteria')
        sns.heatmap(as_data_frame.corr(method=corr_method), cmap='Blues', vmin=-1, vmax=1, xticklabels=False,
                    yticklabels=False)
        plt.title(corr_name + ' correlation bacteria with taxonomy level ' + str(taxonomy_level))
        # plt.savefig(os.path.join(folder, "correlation_heatmap_bacteria.svg"), bbox_inches='tight', format='svg')
        plt.savefig(os.path.join(self.folder, "correlation_heatmap_bacteria.png"))
        # plt.show()
        plt.clf()
        plt.close()
from preprocess_tools import split_accord_to_col, calculate_column_mean

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)



def draw_arrows(d_dicts):
    for df in d_dicts.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        artist = []
        all_color = ["r", "b", "orange", "c", "y", "purple"].__iter__()
        x_lim_val_min, y_lim_val_min, z_lim_val_min = 0, 0, 0
        x_lim_val_max, y_lim_val_max, z_lim_val_max = 0, 0, 0
        for t in d_dicts[df].keys():
            gdm_means = []
            control_means = []
            for pca in ["0", "1", "2"]:
                c_g_d = split_accord_to_col(d_dicts[df][t], "Control_GDM", ["GDM", "Control"])
                gdm_means.append(calculate_column_mean(c_g_d["GDM"], pca))
                control_means.append(calculate_column_mean(c_g_d["Control"], pca))
            x_lim_val_min = min(x_lim_val_min, min(gdm_means[0], control_means[0]))
            x_lim_val_max = max(x_lim_val_max, max(gdm_means[0], control_means[0]))
            y_lim_val_min = min(y_lim_val_min, min(gdm_means[1], control_means[1]))
            y_lim_val_max = max(y_lim_val_max, max(gdm_means[1], control_means[1]))
            z_lim_val_min = min(z_lim_val_min, min(gdm_means[2], control_means[2]))
            z_lim_val_max = max(z_lim_val_max, max(gdm_means[2], control_means[2]))

            a = Arrow3D([0, gdm_means[0]], [0, gdm_means[1]], [0, gdm_means[2]], mutation_scale=20,
                        lw=3, arrowstyle="-|>", color=all_color.__next__(), label=f"GDM {t}")
            ax.add_artist(a)
            b = Arrow3D([0, control_means[0]], [0, control_means[1]], [0, control_means[2]], mutation_scale=20,
                        lw=3, arrowstyle="-|>", color=all_color.__next__(), label=f"Control {t}")
            ax.add_artist(b)
            artist.extend([a, b])
            # plt.tight_layout()
            # ax.plot(xs=[0, gdm_means[0]], ys=[0, gdm_means[1]], zs=[0, gdm_means[2]])
        ax.legend(artist, [i.get_label() for i in artist])
        ax.set_title(df)
        ax.set_xlim(x_lim_val_min - 0.5, x_lim_val_max + 0.5)
        ax.set_ylim(y_lim_val_min - 0.5, y_lim_val_max + 0.5)
        ax.set_zlim(z_lim_val_min - 0.5, z_lim_val_max + 0.5)
        plt.savefig("plots/" + f"{df}.png")
        plt.show()

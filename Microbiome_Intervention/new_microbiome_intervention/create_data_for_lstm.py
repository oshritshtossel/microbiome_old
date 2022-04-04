import os
import pickle
import pandas as pd
import numpy as np

def single_bakteria_time_series(otu_path, folder, time_points):
    df = pd.read_csv(otu_path)
    df = df.iloc[:, 1:]

    for c in df.columns:
        prev_tp = time_points[0]
        samples = []
        sample = []
        for i in range(1, len(time_points)):
            current_tp = time_points[i]
            if current_tp == prev_tp + 1:
                sample.append([list(df.iloc[i - 1, :]), [df.iloc[i, :][c]]])
            elif current_tp == prev_tp + 2:
                mean_full = list((np.array(df.iloc[i - 1, :]) + np.array(df.iloc[i, :])) / 2)
                mean_only = list((np.array(df.iloc[i - 1, :][c]) + np.array([df.iloc[i, :][c]])) / 2)
                sample.append([list(df.iloc[i - 1, :]), mean_only])
                sample.append([mean_full, [df.iloc[i, :][c]]])
            else:
                samples.append(sample)
                sample = []
            prev_tp = current_tp
        samples = [i for i in samples if len(i) > 1]
        final = []
        for sample in samples:
            string_x = ''
            string_y = ''
            for k in range(len(sample)):
                if k > 0:
                    string_x += ';'
                    string_y += ';'
                a = sample[k][0]
                b = sample[k][1]
                string_x += str(a)
                string_y += str(b)
            final.append([string_x, string_y])
        final = pd.DataFrame(final)
        final.columns = ['X', 'Y']
        final.to_csv(folder + c + '.csv',index=False)
def multi_bakteria_time_series(otu_path, folder, time_points):
    df = pd.read_csv(otu_path)
    df = df.iloc[:,1:]

    prev_tp = time_points[0]
    samples = []
    sample = []
    for i in range(1,len(time_points)):
        current_tp = time_points[i]
        if current_tp == prev_tp+1:
            sample.append([list(df.iloc[i-1,:]), list(df.iloc[i,:])])
        elif current_tp == prev_tp+2:
            mean = list((np.array(df.iloc[i-1,:]) + np.array(df.iloc[i,:]))/2)
            sample.append([list(df.iloc[i-1,:]), mean])
            sample.append([mean, list(df.iloc[i,:])])
        else:
            samples.append(sample)
            sample = []
        prev_tp = current_tp
    samples = [i for i in samples if len(i) > 1]
    final = []
    for sample in samples:
        string_x = ''
        string_y = ''
        for k in range(len(sample)):
            if k > 0:
                string_x += ';'
                string_y += ';'
            a = sample[k][0]
            b = sample[k][1]
            string_x += str(a)
            string_y += str(b)
        final.append([string_x, string_y])
    final = pd.DataFrame(final)
    final.columns = ['X', 'Y']
    final.to_csv(folder + 'time_series.csv', index=False)

if __name__ == "__main__":
    otu_path = '../../../PycharmProjects/data_microbiome_in_time/GVHD/OTU_merged_General_task.csv'
    df = pd.read_csv(otu_path)
    time_points = df['ID']
    time_points = [int(i.split('W')[-1]) for i in time_points]
    single_bakteria_time_series(otu_path=otu_path, folder='../../../PycharmProjects/data_microbiome_in_time/GVHD/bacteria_time_series/', time_points=time_points)
    multi_bakteria_time_series(otu_path=otu_path, folder='../../../PycharmProjects/data_microbiome_in_time/GVHD/', time_points=time_points)

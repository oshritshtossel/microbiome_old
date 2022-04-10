from LearningMethods.Regression.regression_in_time import regression
import pandas as pd
import  numpy as np
df = pd.read_csv('../../../PycharmProjects/saliva/data/saliva_full_data.csv')
df.index = df['ID']
meta = pd.read_csv('../../../PycharmProjects/saliva/data/israel_saliva_meta.csv')
meta.index = meta['ID']
gdm_idx = list(meta[meta['Control_GDM'] == 'GDM'].index)
df_xs = pd.read_csv('../../../PycharmProjects/saliva/data/israel_xs.csv')
df_ys = pd.read_csv('../../../PycharmProjects/saliva/data/israel_ys.csv')
## including only gdm for gdm russia
# df_xs = df_xs[df_xs['ID'].isin(gdm_idx)]
# df_ys = df_ys[df_ys['ID'].isin(gdm_idx)]
df_xs = df_xs.iloc[:, 1:]
df_ys = df_ys.iloc[:, 1:]
df = df.iloc[:, 1:]
good_cols = []
for c in df_ys.columns:
    if len(np.unique(df_ys[c])) >= 4:
        good_cols.append(c)
df_ys = df_ys[good_cols]
correlations, mse_means, coeeficents = regression(df, ['Lasso'], df_xs=df_xs, df_ys=df_ys, xsysflag=1)
# coeeficents['Lasso'].to_csv('../../../PycharmProjects/saliva/coeff_russia_lasso_shuffle.csv')
from LearningMethods.Regression.regression_in_time import regression

df = pd.read_csv('../../../PycharmProjects/saliva/data/saliva_full_data.csv')
df.index = df['ID']
df_xs = pd.read_csv('../../../PycharmProjects/saliva/data/russia_xs.csv')
df_ys = pd.read_csv('../../../PycharmProjects/saliva/data/russia_ys.csv')
df_xs = df_xs.iloc[:, 1:]
df_ys = df_ys.iloc[:, 1:]
df = df.iloc[:, 1:]
good_cols = []
for c in df_ys.columns:
    if len(np.unique(df_ys[c])) >= 4:
        good_cols.append(c)
df_ys = df_ys[good_cols]
correlations, mse_means, coeeficents = regression(df, ['Lasso'], df_xs=df_xs, df_ys=df_ys, xsysflag=1)
coeeficents['Lasso'].to_csv('../../../PycharmProjects/saliva/coeff_russia_lasso.csv')
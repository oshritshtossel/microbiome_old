import pickle
import random
import numpy as np
from scipy.stats import stats, spearmanr
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression, LogisticRegression, BayesianRidge, Lasso
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
svr_default_params = {'kernel': 'linear', 'gamma': 'auto', 'C': 0.1}

decision_tree_default_params = {"max_features": 2, "min_samples_split": 4,
                                "n_estimators": 50, "min_samples_leaf": 2}

random_forest_default_params ={"max_depth": 5, "min_samples_split": 4,
                                "n_estimators": 50, "min_samples_leaf": 2}

# -------- general learning functions --------
def get_model(model_name, model_params={}):
    if model_name == 'Linear_regression':
        return LinearRegression()
    elif model_name=='Lasso':
        if model_params=={}:
            return Lasso(alpha=0.1)
        else:
            return Lasso(alpha=model_params['alpha'])
    elif model_name=='Ridge':
        if model_params == {}:
            return Ridge(alpha=1.0)
        else:
            return Ridge(alpha=model_params['alpha'])
    elif model_name=='ARD':
        if model_params == {}:
            return ARDRegression()
        else:
            return ARDRegression(alpha_1=model_params['alpha'], lambda_1=model_params['lambda'])
    elif model_name == 'SVR':
        if model_params == {}:
            return svm.SVR(C=1e-1, epsilon=0.1, kernel='linear', gamma='auto')
        else:
            return svm.SVR(C=model_params['c'], epsilon=model_params['epsilon'], kernel='linear', gamma='auto')
    else:
        print('model not supported')
        raise -1
# main learning loop - same for all basic algorithms
def learning_cross_val_loop(model,x, y):
    kf = KFold(n_splits=5, shuffle=True)
    preds = []
    reals = []
    coef = np.zeros(x.shape[1])
    # Split the data set
    for train_index, test_index in kf.split(x):
        clf = get_model(model)
        train_x, test_x = x.iloc[train_index,:], x.iloc[test_index,:]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]
        # FIT
        clf.fit(train_x, train_y)
        # GET RESULTS
        coef += clf.coef_.flatten()

        y_pred = clf.predict(test_x)
        reals.append(test_y)
        preds.append(y_pred)
    coef = coef / kf.n_splits
    spearmans = []
    p_values = []
    test_mses = []
    for pred, real in zip(preds, reals):
        spearman, p_value = stats.spearmanr(real, pred)
        test_mse = mean_squared_error(real, pred)
        spearmans.append(spearman)
        p_values.append(p_value)
        test_mses.append(test_mse)
    spearmans = np.array(spearmans)[~np.isnan(np.array(spearmans))]
    spearman = np.mean(spearmans)
    # p_value = np.mean(np.array(p_values))
    test_mse = np.mean(np.array(test_mses))
    return spearman, test_mse, coef


def calc_corr_on_joined_results(y_test, y_pred):
    tesr_spearman, test_p_value = stats.spearmanr(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    mixed_rho, mixed_pvalues, mixed_rmse = calc_evaluation_on_mix_predictions(y_test, y_pred)

    return tesr_spearman, test_p_value, mixed_rho, mixed_pvalues, test_mse, mixed_rmse


def calc_evaluation_on_mix_predictions(y_pred, y_test):
    mixed_y_list, mixed_rhos, mixed_pvalues, mixed_rmse = [], [], [], []
    for num in range(10):  # run 10 times to avoid accidental results
        mixed_y_fred = y_pred.copy()
        random.shuffle(mixed_y_fred)
        mixed_y_list.append(mixed_y_fred)
        rho_, pvalue_ = spearmanr(y_test, mixed_y_fred, axis=None)
        mixed_rhos.append(rho_)
        mixed_pvalues.append(pvalue_)
        mixed_rmse.append(mean_squared_error(y_test, mixed_y_fred))
    return np.array(mixed_rhos).mean(), np.array(mixed_pvalues).mean(), np.array(mixed_rmse).mean()


def calc_spearmanr_from_regressor(reg, X_test, y_test):
    b_1 = reg.coef_
    b_n = reg.intercept_

    # use the b value to decide with bacteria have influence on the tested bacteria
    y_pred = []
    for x in X_test:
        reg_y = np.dot(x, b_1) + b_n
        y_pred.append(reg_y)

    # check if the bacteria change can be predicted according to the spearman correlation
    rho, p_val = spearmanr(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    # check if the predictions are random
    mixed_rho, mixed_pvalues, mixed_rmse = calc_evaluation_on_mix_predictions(y_pred, y_test)

    return rho, p_val, b_1, mixed_rho, mixed_pvalues, rmse, mixed_rmse, y_pred


# -------- many kind of regressors implementations --------
def calc_linear_regression(x, y):
    clf = LinearRegression()
    spearman, mse = learning_cross_val_loop(clf, x, y)
    return spearman, mse, clf.coef_


def calc_ridge_regression(X_train, X_test, y_train, y_test):
    reg = Ridge().fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


def calc_ard_regression(X_train, X_test, y_train, y_test):
    reg = ARDRegression().fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


def calc_lasso_regression(X_train, X_test, y_train, y_test):
    reg = linear_model.Lasso(alpha=0.01).fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


def calc_bayesian_ridge_regression(X_train, X_test, y_train, y_test):
    reg = BayesianRidge().fit(X_train, y_train)
    reg.score(X_train, y_train)
    return calc_spearmanr_from_regressor(reg, X_test, y_test)


def svr_regression(X_train, X_test, y_train, y_test, params=svr_default_params):
    clf = svm.SVR(C=params['C'], kernel=params['kernel'], gamma=params['gamma'])
    y_test, y_test_pred = learning_cross_val_loop(clf, X_train, X_test, y_train, y_test)
    b_1 = clf.coef_[0]
    rho, pvalue, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse = calc_corr_on_joined_results(y_test, y_test_pred)
    return rho, pvalue, b_1, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse, y_test_pred


def decision_tree_regressor(X_train, X_test, y_train, y_test, params=decision_tree_default_params):
    clf = DecisionTreeRegressor(max_features=params["max_features"], min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"])

    y_test, y_test_pred = learning_cross_val_loop(clf, X_train, X_test, y_train, y_test)

    rho, pvalue, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse = calc_corr_on_joined_results(y_test, y_test_pred)
    return rho, pvalue, "    ", mixed_rho, mixed_pvalues, test_rmse, mixed_rmse, y_test_pred


def random_forest_regressor(X_train, X_test, y_train, y_test, params=random_forest_default_params):

    clf = RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                min_samples_split=params["min_samples_split"], min_samples_leaf=params["min_samples_leaf"])

    y_test, y_test_pred = learning_cross_val_loop(clf, X_train, X_test, y_train, y_test)

    rho, pvalue, mixed_rho, mixed_pvalues, test_rmse, mixed_rmse = calc_corr_on_joined_results(y_test, y_test_pred)
    return rho, pvalue, "    ", mixed_rho, mixed_pvalues, test_rmse, mixed_rmse, y_test_pred

def plot_bar_regression_models():
    with open('d.pickle', 'rb') as handle:
        b = pickle.load(handle)
    fig1, ax =  plt.subplots()
    plt.bar(np.arange(5),[i[1] for i in b['mucusitis'][0].items()], color='cyan', tick_label=['SVR', 'ARD', 'Lasso', 'Ridge', 'Linear Regression'],width=0.4)
    array, prev_maximum, maximum = fix_array([i[1] for i in b['Allergy'][0].items()], [i[1] for i in b['mucusitis'][0].items()])
    bottom = [prev_maximum[i] if maximum[i] - prev_maximum[i] > 0 else 0 for i in range(len(array))]
    plt.bar(np.arange(5),array,bottom=bottom, color='red', tick_label=['SVR', 'ARD', 'Lasso', 'Ridge', 'Linear Regression'],width=0.4)
    array, prev_maximum, maximum = fix_array([i[1] for i in b['VitamineA'][0].items()], maximum)
    bottom = [prev_maximum[i] if maximum[i] - prev_maximum[i] > 0 else 0 for i in range(len(array))]
    bottom[1] = [i[1] for i in b['Allergy'][0].items()][1]
    array[1] =array[1]- bottom[1]
    plt.bar(np.arange(5),array, bottom=bottom, color='pink', tick_label=['SVR', 'ARD', 'Lasso', 'Ridge', 'Linear Regression'],width=0.4)
    array, prev_maximum, maximum = fix_array([i[1] for i in b['GDM'][0].items()], maximum)
    bottom = [prev_maximum[i] if maximum[i] - prev_maximum[i] > 0 else 0 for i in range(len(array))]
    plt.bar(np.arange(5),array,bottom=bottom, color='orange', tick_label=['SVR', 'ARD', 'Lasso', 'Ridge', 'Linear Regression'],width=0.4)
    plt.title('correlation by model and dataset')
    plt.ylabel('Correlation')
    plt.xlabel('Regression Model')
    plt.show()
    fig1.savefig('correlation_reg_models.pdf')

def fix_array(array, prev_maximum):
    maximum = []
    for i in range(len(array)):
        if array[i] > prev_maximum[i]:
            maximum.append(array[i])
            array[i] -= prev_maximum[i]
        else:
            maximum.append(prev_maximum[i])
    return array, prev_maximum, maximum
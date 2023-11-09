import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import scipy
import hncUtility as hnc

if __name__ == '__main__':

    print('neuron correlation analysis started...')
    df = pd.read_csv(r'H:\my research\SWC datasets files\all\merged_all.csv')

    if hasattr(df, 'neuron_name'):
        print('removed neuron_name')
        df.drop('neuron_name', inplace=True, axis=1)

    # our Y parameter
    dependent_variable = "ap_turnover"
    # our X parameter
    predictor_variable = "neurite_lengths_NeuriteType_axon_mean"

    df = df.filter(regex=(dependent_variable + "|" + predictor_variable))
    # df = df[df.columns.drop(list(df.filter(regex=("axon|presynaptic|Axon"))))]
    csv_field_names = df.columns

    scale = preprocessing.StandardScaler()
    df[csv_field_names] = scale.fit_transform(df[csv_field_names].as_matrix())

    for parameter in csv_field_names:
        num_out = 0
        iterator = range(len(df[str(parameter)]))
        for value in iterator:
            if abs(df[str(parameter)][value]) >= 2:
                print('Found outlier:' + str(df[str(parameter)][value]))
                df[str(parameter)][value] = float('nan')
                num_out = num_out + 1
        print('found ' + str(num_out) + ' outliers in ' + str(parameter))

    print(df)

    df=df[df[predictor_variable].notnull()]
    df=df[df[dependent_variable].notnull()]

    print(df)

    x = df[predictor_variable]
    y = df[dependent_variable]

    neuron_model = ols(dependent_variable + " ~ " + predictor_variable, data=df).fit()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=5)

    olsmod = sm.OLS(y_train, x_train)
    olsres = olsmod.fit()
    print(olsres.summary())

    ypred = olsres.predict(x_train)
    ynewpred = olsres.predict(x_test)  # predict out of sample

    print('Fit a model x_train, and calculte MSE with y_train: ', np.mean((y_train - ypred) ** 2))
    print('Fit a model x_train, and calculate MSE with x_test, y_test: ', np.mean((y_test - ynewpred) ** 2))

    fig = hnc.regression_plot(x, y)
    plt.show(fig)




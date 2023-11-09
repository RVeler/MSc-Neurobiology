import math
import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing, linear_model
import pylab
import statsmodels.api as sm
from statsmodels.formula.api import ols


# filter just the fields that contain the names that you want. personal_fields is an array with your full/partial text of the field that will be used
def filter_fields(personal_fields, data_frame):
    print('filter_fields')

    # remove neuron_name because it is a text field
    if hasattr(data_frame, 'neuron_name'):
        print('removed neuron_name' )
        data_frame.drop('neuron_name' ,inplace=True,axis=1)


    dtypes = data_frame.axes
    csv_field_names = dtypes[1]._data

    for existing_field in csv_field_names:
        print('analyzing field: ', existing_field)
        found_field = False
        for my_field in personal_fields:
            # check if my field is inside the existing field
            num = 0
            for my in my_field:
                if existing_field.find(my) >= 0:
                    num = num+1
            if num == len(my_field):
                print('found relevant field name: ' + existing_field)
                found_field = True
            # else:
            #     print('removed field from analysis: ' + existing_field)

        if found_field == False:
            data_frame.drop(existing_field, inplace=True, axis=1)
            print('removed field from analysis: ' + existing_field)


    return data_frame

# plot linear regression of 2 parameters arrays (x,y). args - this is for insert the q value of the FDR and not the original p value
def regression_plot(x,y, args = None):

    fig, ax = plt.subplots()
    fit = np.polyfit(x, y, deg=1)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)

    show = 'r = ' + str(round(r_value,2))
    if args != None:
        show2 = 'q = ' + str(round(args,5))
    else:
        show2 = 'p = ' + str(round(p_value,5))
    if args == 'AD':
        show = ' '
    else:
        ax.plot(x ,fit[0] * x + fit[1], color='red', label='Outline label')
    ax.scatter(x, y)
    plt.xlabel(x.name.replace('_', ' '))
    plt.ylabel(y.name.replace('_', ' '))
    plt.figtext(.15, .8, show)
    plt.figtext(.15,.75, show2)
    #pylab.ylim([-2,2])
    #pylab.xlim([-2, 2])
    print('std: ' + str(std_err))
    print('intercept: ' + str(intercept))
    file_name = x.name + '_' + y.name +'.pdf'
    #insert here the path you want the figure will be saved
    plt.savefig(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\correlation fdr 0.05\\' + file_name, format='pdf')
    return

#removes outliers from a Dataframe:

def remove_outliers(df):

    csv_field_names = df.columns
    ol = []
    for parameter in csv_field_names:
        num_out = 0
        iter = range(len(df[str(parameter)]))
        stand = df[str(parameter)].std()
        mean_par = np.mean(df[str(parameter)])
        for value in iter:
            if abs(df[str(parameter)][value] - mean_par) >= stand * 2: #insert here the threshold std you want for outliers
                print('Found outlier:' + str(df[str(parameter)][value]))
                df[str(parameter)][value] = float('nan')
                num_out = num_out + 1
        print('found ' + str(num_out) + ' outliers in ' + str(parameter))
        ol.append(num_out)
    return df, ol

# arranges the relevant correlations into csv file

def arrange_corr_csv(correlations,p_correlations, rsqu_correlations):
    correlation_column_labels = ['Correlation', 'p_value', 'Adj_rsquared', 'Field_A', 'Field_B']
    rdf = pd.DataFrame(columns=correlation_column_labels)
    minimal_correlation_value = 0
    max_p_value = 1

    for x_param in correlations.columns:
        for y_param in correlations.columns:
            corr_value = correlations.loc[x_param, y_param]
            p_value = p_correlations.loc[x_param, y_param]
            r_value = rsqu_correlations.loc[x_param, y_param]
            if abs(corr_value) >= minimal_correlation_value and p_value <= max_p_value:
                # skipping save variables
                if x_param != y_param:
                    # look for duplicate existing value
                    existing_var = rdf[(rdf.Field_A == y_param) & (rdf.Field_B == x_param)]
                    if existing_var.empty:
                        rdf = rdf.append({'Correlation': corr_value, 'p_value': p_value, 'Adj_rsquared': r_value, 'Field_A': x_param, 'Field_B': y_param},
                                         ignore_index=True)

    # save to a csv file
    file_name = r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\rat L5\trial\correlation.csv'
    rdf.to_csv(file_name, sep=',')
    print('saved to file: ', file_name)

    return file_name

# create new p value matrix with FDR correction
def fdr(file, p_matrix, op = 'None'): # op = v: create a matrix for 2 colors (significant and not significant), op = red: create a colormap matrix in which we dont have the white color in the colormap
    df = pd.read_csv(file)
    Field_A = list(df['Field_A'])
    Field_B = list(df['Field_B'])
    iter = range(len(Field_A))
    for it in iter:
        x = Field_A[it]
        y = Field_B[it]
        p = df['fdr'][it]
        if op == 'v':
            if p > 0.01: #write here the q threshold
                p_matrix.at[str(x), str(y)] = 0.5
                p_matrix.at[str(y), str(x)] = 0.5
            else:
                p_matrix.at[str(x), str(y)] = 0
                p_matrix.at[str(y), str(x)] = 0
            p_matrix.at[str(x), str(x)] = 0.5
            p_matrix.at[str(y), str(y)] = 0.5
        elif op == 'red':
            if p < 0.000001:
                p_matrix.at[str(x), str(y)] = 0.000001
                p_matrix.at[str(y), str(x)] = 0.000001
            else:
                p_matrix.at[str(x), str(y)] = p
                p_matrix.at[str(y), str(x)] = p
                p_matrix.at[str(x), str(x)] = 0.000001
                p_matrix.at[str(y), str(y)] = 0.000001
        else: # op = None: create a regular matrix for q colormap
            p_matrix.at[str(x), str(y)] = p
            p_matrix.at[str(y), str(x)] = p
            p_matrix.at[str(x), str(x)] = 0.5
            p_matrix.at[str(y), str(y)] = 0.5
    return p_matrix


#claculates (by train and test procedure) and return the MSE ratio between 2 data sets, if op = v: return a list of (mse1, mse2, N(1), N(2)
def mse_ratio(df,dft,dependent_variable,predictor_variable, op = None):

    df = df.filter(regex=(dependent_variable + "|" + predictor_variable))
    dft = dft.filter(regex=(dependent_variable + "|" + predictor_variable))
    csv_field_names = df.columns

    df=df[df[predictor_variable].notnull()]
    df=df[df[dependent_variable].notnull()]

    dft = dft[dft[predictor_variable].notnull()]
    dft = dft[dft[dependent_variable].notnull()]

    if len(dft[predictor_variable]) == 0 or len(dft[dependent_variable]) == 0:
        return ['Nan', 'Nan', 'Nan', 'Nan']

    scale = preprocessing.StandardScaler()
    df[csv_field_names] = scale.fit_transform(df[csv_field_names].as_matrix())
    dft[csv_field_names] = scale.fit_transform(dft[csv_field_names].as_matrix())

    x_train = df[predictor_variable]
    y_train = df[dependent_variable]

    x_test = dft[predictor_variable]
    y_test = dft[dependent_variable]

    neuron_model = ols(dependent_variable + " ~ " + predictor_variable, data=df).fit()

    olsmod = sm.OLS(y_train, x_train)
    olsres = olsmod.fit()

    ypred = olsres.predict(x_train)
    ynewpred = olsres.predict(x_test)  # predict out of sample

    mse1 = np.mean((y_train - ypred) ** 2)
    mse2 = np.mean((y_test - ynewpred) ** 2)
    if op == 'v':
        return [mse1, mse2, len(ypred), len(ynewpred)]
    elif mse2 == 0 and mse1 == 0:
        return 1
    elif mse2 == 0:
        return 2
    else:
        return mse1/mse2





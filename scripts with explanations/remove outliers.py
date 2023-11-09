import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


if __name__ == '__main__':

#write below the path of the matrix you want to mark the outliers
    df = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\merged_alldatset!!!.csv')
    if hasattr(df, 'neuron_name'):
        print('removed neuron_name')
        df.drop('neuron_name', inplace=True, axis=1)

    csv_field_names = df.columns

    for parameter in csv_field_names:
        num_out = 0
        iter = range(len(df[str(parameter)]))
        stand = df[str(parameter)].std()
        mean_par = np.mean(df[str(parameter)])
        for value in iter:
            if abs(df[str(parameter)][value] - mean_par) >= stand * 2: # here, the number is the std threshold you set.
                print('Found outlier:' + str(df[str(parameter)][value]))
                df[str(parameter)][value] = '^' + str(df[str(parameter)][value]) # '^' in the csv file represent outlier.
                num_out = num_out + 1
        print('found ' + str(num_out) + ' outliers in ' + str(parameter))
    #write below the [ath and file name for the df with marked outliers
    file_path = r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\outliers.csv'
    df.to_csv(file_path)




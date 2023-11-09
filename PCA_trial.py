import numpy, math
import hncUtility as hnc
import csv
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def onclick(event):
    if event.inaxes is not None:
        minx = int(math.floor(round(event.xdata)))
        miny = int(math.floor(round(event.ydata)))
        print('event.xdata ', minx, ' event.ydata ', miny)
        print('neuron parameters: X=', names[minx], ' ; Y=', names[miny])



if __name__ == '__main__':
    indexname = 'hnc1'

    print('PCA test.')
    fields = ['bifur']

    df = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\merged_alldatset! reduce.csv')

    if hasattr(df, 'neuron_name'):
        print('removed neuron_name' )
        df.drop('neuron_name' ,inplace=True,axis=1)

    csv_field_names = df.columns

    pd.set_option('display.width', 1000)
    pd.set_option('precision', 2)

    #hnc.remove_outliers(df)


    dtypes = df.axes

    #df = df.fillna(0).reset_index()
    # these are the field names from the CSV file
    names = dtypes[1]._data

    scale = preprocessing.StandardScaler()
    df[names] = scale.fit_transform(df[names].as_matrix())



    array = df.values
    X = array[:, 0:100]
    Y = array[:, 8]

    # feature extraction
    pca = PCA(n_components=2)
    X_r = pca.fit(array).transform(array)
    X_f = pca.fit(array)

    # summarize components
    Print = "Explained Variance: " + str(X_f.explained_variance_ratio_)
    comp = X_f.components_
    print(Print)
    print(X_f.components_[0])

    plt.figure()

    lw = 2
    j = range(0, len(Y))

    plt.scatter(X_r[:,0], X_r[:,1], color='navy', alpha=.8, lw=lw, label='cells')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Lior dataset')
    plt.show()

    df2 = pd.DataFrame.from_records(comp, columns=df.columns)

    filename = r'E:\my research\pca_analysis.csv'

    df2.to_csv(filename, index=False)


    print('saved data to: ', r'E:\my research\pca_analysis.csv')

    #merged = df2.merge(df, on=df.columns)
    #merged.to_csv(r'E:\my research\pca_analysis2.csv', index=False)

    # plt.subplot(111)
    # plt.scatter(values[:, 0], values[:, 1], c=y_pred)
    # plt.title("Neurom KMeans Test")
    # plt.show()

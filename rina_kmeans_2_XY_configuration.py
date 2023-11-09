import numpy, math

from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import hncUtility

def onclick(event):
    if event.inaxes is not None:
        minx = int(math.floor(round(event.xdata)))
        miny = int(math.floor(round(event.ydata)))
        print('event.xdata ', minx, ' event.ydata ', miny)
        print('neuron parameters: X=', names[minx], ' ; Y=', names[miny])


if __name__ == '__main__':
    indexname = 'hnc1'

    print('kmeans computation test.')
    my_fields = [['remote_bifurcation_angles_NeuriteType_apical_dendrite_mean'], ['total_length_NeuriteType_all_value'], ['number_of_terminations_NeuriteType_axon_value'], ['number_of_sections_per_neurite_NeuriteType_axon_max']]

    df = pd.read_csv('E:\my research\SWC datasets files\datasetall.csv')

    df = hncUtility.filter_fields(my_fields, df)

    pd.set_option('display.width', 1000)
    pd.set_option('precision', 3)

    dtypes = df.axes

    df = df.fillna(0).reset_index()
    # these are the field names from the CSV file
    names = dtypes[1]._data

    # remove this scaling part if you find it hard to identify the data - it is being scaled...
    scale = preprocessing.StandardScaler()
    df[names] = scale.fit_transform(df[names].as_matrix())

    values = df.values
    y_pred = KMeans(n_clusters=3).fit_predict(values)

    plt.subplot(111)
    plt.grid(True)
    #values[:, 0] column has just row number
    #values[:, 1] contains the 1st column data(from the left), which is 'number_of_bifurcations_NeuriteType_apical_dendrite_value'
    #values[:, 5] should be 'number_of_forking_points_NeuriteType_apical_dendrite_value'
    plt.scatter(values[:, 1], values[:, 0], c=y_pred)
    plt.title("3 KMeans Cluster. number_of_bifurcations_NeuriteType_apical_dendrite_value By number_of_forking_points_NeuriteType_apical_dendrite_value")
    plt.show()
    plt.savefig




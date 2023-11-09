import neurom as nm
import numpy
import numpy as np
import os
import os.path
import csv
import sys
from pandas import merge, read_table
import pandas as pd

if __name__ == '__main__':

    from pandas import merge, read_table

    # neuron_csv = read_table('/Users/user/brain/python1/HNC1/csv/full_10.csv', sep=r'[,]', header=None)
    # sp_csv = read_table('/Users/user/brain/python1/HNC1/csv/rina_dataset_1.csv', sep=r'[,]',  header=None)

    # a - put the path of the SP matrix dataset
    #b - put the path of the neuroanatomy dataset


    a = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\trial\merge.csv')
    b = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\trial\mergedtrial.csv')

    merged = a.merge(b, on='neuron_name',how='left')
    #here you put the path and name that you want the merged data will be saved
    merged.to_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\trial\merged_trial.csv', index=False)

    # merged = merge(neuron_csv, sp_csv, how='left', on='neuron_name')
    # merged.to_csv('/Users/user/brain/python1/HNC1/csv/merged.csv', index=False)


    print('done.')

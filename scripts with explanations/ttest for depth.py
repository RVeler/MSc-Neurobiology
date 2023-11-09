import pandas as pd
import hncUtility as hnc
import numpy
from scipy import stats

if __name__ == '__main__':

#here you put the path of the file depth_groups
    df = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\depth_groups.csv')

    if hasattr(df, 'neuron_name'):
        print('removed neuron_name')
        df.drop('neuron_name', inplace=True, axis=1)

    csv_field_names = df.columns

    pd.set_option('display.width', 1000)
    pd.set_option('precision', 2)

    hnc.remove_outliers(df)

    col = ['1 - mean +/- std', '2 - mean +/- std', 'ttest p value']
    ind = csv_field_names[1:]
    rdf = pd.DataFrame(columns=col, index=ind)
    for i in ind:
        for c in col:
            g1 = df.loc[df['Soma_depth'] == 1, i]
            g2 = df.loc[df['Soma_depth'] == 2, i]
            if c == '1 - mean +/- std':
                rdf.at[i,c] = str(round(numpy.mean(g1),2)) + ' +/- ' + str(round(g1.std(),2))
            elif c == '2 - mean +/- std':
                rdf.at[i,c] = str(round(numpy.mean(g2),2)) + ' +/- ' + str(round(g2.std(),2))
            else:
                rdf.at[i,c] = stats.ttest_ind(g1, g2, nan_policy='omit')[1]

#here you put the path of the file matrix (DataFrame rdf) you want to save
    file_name = r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\ttest by depth.csv'
    rdf.to_csv(file_name, sep=',')



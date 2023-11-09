import pandas as pd
import hncUtility as hnc
import numpy as np
from matplotlib import pyplot as plt, cm

if __name__ == '__main__':

    # here you put the path of our dataset matrix
    df = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\merged_alldatset! reduce.csv')
    # here you put the path of the other dataset matrix
    dft = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\Layer 2-3 PyrNs rat\mergedratL23 reduce.csv')
    #here you put the path of fdr values of our dataset ('FDRallreduce')
    fdr = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\FDRallreduce.csv')

    if hasattr(df, 'neuron_name'):
        print('removed neuron_name')
        df.drop('neuron_name', inplace=True, axis=1)
    if hasattr(dft, 'neuron_name'):
        print('removed neuron_name')
        dft.drop('neuron_name', inplace=True, axis=1)


    hnc.remove_outliers(df)
    hnc.remove_outliers(dft)

    choice = input('press 1 for colormap or 2 for dataframe:') #in the command window you will be asked to write 1 or 2 for colormap of MSE ratio values or dataframe with MSE values
    if choice == '1':
        cols = df.columns[:]
        mse_map = pd.DataFrame(index=cols, columns=cols, dtype='float')
        for ind in cols:
            for col in cols:
                mse_ratio = hnc.mse_ratio(df, dft, ind, col)
                mse_map.at[str(ind), str(col)] = mse_ratio

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(mse_map, vmin=0, vmax=2, cmap=cm.coolwarm, picker=True)
        fig.colorbar(cax)
        ticks = np.arange(0, len(cols), 1)
        nums = [x for x in range(1, 48) if x not in [1, 2, 3, 4, 5, 6, 7, 13, 20, 26, 41]]
        # nums = np.array(indices)+1
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(nums, {'fontsize': 4})
        ax.set_yticklabels(nums, {'fontsize': 4})
        ax.tick_params(labeltop=True, labelbottom=True)
       # write below the path and name you want the figure will be saved
        plt.savefig(
            r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\Layer 2-3 PyrNs Human\trial\\' + 'mse humanL23' + '.pdf',
            dpi=200)
    else:
        column_labels = ['N(mse2)','mse2', 'N(mse1)', 'mse1', 'Field_B', 'Field_A'] # N represent the number of the neurons in those parameters. 1 is for our dataset and 2 is for the other dataset.
        rdf = pd.DataFrame(columns=column_labels)
        Field_A = list(fdr['Field_A'])
        iter = range(len(Field_A))
        for it in iter:
            if fdr['fdr'][it] <= 0.01 and hasattr(df, fdr['Field_A'][it]) and hasattr(df, fdr['Field_B'][it]):
                mse_ratio = hnc.mse_ratio(df, dft, fdr['Field_A'][it], fdr['Field_B'][it], op='v')
                rdf = rdf.append(
                    {'N(mse2)': mse_ratio[3], 'N(mse1)': mse_ratio[2], 'mse2': mse_ratio[1], 'mse1': mse_ratio[0], 'Field_A': fdr['Field_A'][it],
                     'Field_B': fdr['Field_B'][it]},
                    ignore_index=True)
        # write below the path and name you want the featuremap will be saved
        file_name = r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\Layer 2-3 PyrNs rat\mse data ratL23.csv'
        rdf.to_csv(file_name, sep=',')

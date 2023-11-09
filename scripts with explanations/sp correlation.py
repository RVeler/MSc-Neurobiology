import numpy as np, math
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt, cm
from matplotlib import colors
import pandas as pd
import scipy
from sklearn import preprocessing
import hncUtility as hnc


if __name__ == '__main__':

    print('correlation plot test')

    df = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\sp correlation\sp for corr.csv')

    if hasattr(df, 'neuron_name'):
        print('removed neuron_name')
        df.drop('neuron_name', inplace=True, axis=1)


    pd.set_option('display.width', 1000)
    pd.set_option('precision',2)


    hnc.remove_outliers(df)

    depth = ['soma_depth', '_depth', '_distance']
    depend = ['_turnover', '_density']
    type = ['bas', 'ap', 'ax']

    for typ in type:
        depth = ['soma_depth', typ+'_depth', typ+'_distance']
        depend = [typ+'_turnover', typ+'_density']
        correlations = pd.DataFrame(index=depend, columns=depth, dtype='float')
        correlations_p = pd.DataFrame(index=depend, columns=depth, dtype='float')
        for ind in depth:
          for col in depend:
            df3 = df
            df3 = df3[df3[ind].notnull()]
            df3 = df3[df3[col].notnull()]
            fit = np.polyfit(df3[ind], df3[col], deg=1)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df3[ind], df3[col])
            correlations.at[str(ind), str(col)] = r_value
            correlations_p.at[str(ind), str(col)] = p_value
            hnc.regression_plot(df3[ind], df3[col])

       # creating r color map
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap=cm.coolwarm, picker=True)
        fig.colorbar(cax)
        ticks = np.arange(1, len(depth), 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(depth, {'fontsize': 11})
        ax.set_yticklabels(depend, {'fontsize': 11})
      # creating p color map
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        cax2 = ax2.matshow(correlations_p, norm=colors.LogNorm(vmin=0.1**7, vmax=0.1**0), cmap=cm.coolwarm_r, picker=True)
        fig2.colorbar(cax2)
        ax2.set_xticks(ticks)
        ax2.set_yticks(ticks)
        ax2.set_xticklabels(depth, {'fontsize': 11})
        ax2.set_yticklabels(depend, {'fontsize': 11})
    plt.show()



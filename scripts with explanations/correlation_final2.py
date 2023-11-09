import numpy as np, math
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt, cm
from matplotlib import colors
import pandas as pd
import scipy
from sklearn import preprocessing
import hncUtility as hnc
from scipy.stats import ttest_ind

def onclick(event):
    if event.inaxes is not None:
        minx = int(math.floor(round(event.xdata)))
        miny = int(math.floor(round(event.ydata)))
        print('event.xdata ', minx, ' event.ydata ', miny)
        print('neuron parameters: X=', csv_field_names[minx], ' ; Y=', csv_field_names[miny])
        print('correlation value: ', correlations.iloc[minx][miny], ' p_value: ', correlations_p.iloc[minx][miny] )


if __name__ == '__main__':

    print('correlation plot test')

    #write below the path of the matrix of the dataset you want to work with.
    df = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\merged_alldatset! reduce.csv')

    if hasattr(df, 'neuron_name'):
        print('removed neuron_name')
        df.drop('neuron_name', inplace=True, axis=1)

    csv_field_names = df.columns

    pd.set_option('display.width', 1000)
    pd.set_option('precision',2)

    #remove outliers (set the threshold in the hncUtility script)
    hnc.remove_outliers(df)


    indices = range(len(csv_field_names))
    cols = df.columns[:]
    #create correlation matrices for corellation(r values), pvalues, r square values, number of neurons in each correlation,
    #pfdr values, respectively.
    correlations = pd.DataFrame(index=cols, columns=cols, dtype='float')
    correlations_p = pd.DataFrame(index=cols, columns=cols, dtype='float')
    correlations_rsqu = pd.DataFrame(index=cols, columns=cols, dtype='float')
    correlations_num = pd.DataFrame(index=cols, columns=cols, dtype='float')
    correlations_pfdr = pd.DataFrame(index=cols, columns=cols, dtype='float')

    for ind in cols:
        for col in cols:
            df3 = df
            df3 = df3[df3[ind].notnull()]
            df3 = df3[df3[col].notnull()]
            if len(df3[ind]) > 0 and len(df3[col]) > 0:
                fit = np.polyfit(df3[ind], df3[col], deg=1)
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df3[ind], df3[col])
                neuron_model = ols(str(ind) + "~" + str(col), missing='drop', data=df3).fit()
            #coef = neuron_model.params[1]
                pval = neuron_model.pvalues[1]
                nval = len(df3[ind])
                r_square = neuron_model.rsquared_adj
                correlations.at[str(ind), str(col)] = r_value
                #if ind == 'AD':
                #pvalue = ttest_ind(df3.loc[df[ind] == 0, str(col)],df3.loc[df[ind] == 1, str(col)])[1]
                #elif col == 'AD':
                #p_value = ttest_ind(df3.loc[df[col] == 0, str(ind)],df3.loc[df[col] == 1, str(ind)])[1]
                correlations_p.at[str(ind), str(col)] = p_value
                correlations_rsqu.at[str(ind), str(col)] = r_square
                correlations_num.at[str(ind), str(col)] = nval
            else:
                correlations.at[str(ind), str(col)] = float('nan')
                correlations_p.at[str(ind), str(col)] = float('nan')
                correlations_rsqu.at[str(ind), str(col)] = float('nan')
  #create the p_fdr matrix with the relevant values that Nitza sent us. you can add 'v' or 'red' in the end for another conditions (see detailes in the hnc script)
    p_fdr = hnc.fdr(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\FDRallreduce.csv', correlations_pfdr, 'v')

  # remove the '#' sign below in order to save the relevant matrices

    #p_fdr.to_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\rat L5\trial\p_fdr.csv')
    #correlations_p.to_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\correlations_p-r.csv', sep=',')
    #correlations_num.to_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\rat L5\trial\correlations_n-r.csv', sep=',')

# remove the "#" sign for the relevant 'file' you want for correlation figures:
    #file = hnc.arrange_corr_csv(correlations, correlations_p, correlations_rsqu) # - for original p_values
    file = r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\fdr_correction.csv' # - for fdr values (you need to put the path for Nitza fdr values matrix of the relevant dataset)

#here there is a code for looping the correlations amd creating correaltion figures for significant correlations.
    df2 = pd.read_csv(file)
    Field_A = list(df2['Field_A'])
    Field_B = list(df2['Field_B'])
    fdr = list(df2['fdr']) # use it when you use q values and not p
    iter = range(len(Field_A))
    for it in iter:
        x = Field_A[it]
        y = Field_B[it]
        z = fdr[it] # use it when you use q values and not p
        #r = df2['Adj_rsquared'][it]
        #p = df2['p_value'][it] # use it when you use p values and not q
        df3 = df
        df3 = df3[df3[x].notnull()]
        df3 = df3[df3[y].notnull()]
        # write below the relevant q(z) or p(p) values you want to create correlation figures
        if z <= 0.01:
            hnc.regression_plot(df3[x],df3[y],z) # z is optional for using fdr q values. you insert the path you want to save in it the figures, in the hnc script.

# here we create colormap figures for r and p\q values of correlations.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap=cm.coolwarm, picker=True)
    fig.colorbar(cax)


    # ticks is an array, created by numpy, with the number of csv fields, stepped by 1, starting from 0
    ticks = np.arange(0, len(cols), 1)
    ticks2 = [0, 4, 9, 14, 19,24, 29,34] # use it for reduced parameters
    ticks3 = [0, 4, 9, 14, 19,24, 29,34, 39, 44, 49, 54]
    nums = [x for x in range(1,48) if x not in [1,2,3,4,5,6,7,13,20,26,41]] # use it for reduced parameters
    nums2 = [nums[i] for i in ticks2]
    #nums = np.array(indices)+1
    #nums = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] # use it for all parameters
    ax.set_xticks(ticks, minor = True)
    ax.set_yticks(ticks, minor = True)
    #plt.xticks(ticks3, nums, fontsize=8) #use this line and the kine below for all parameters
    #plt.yticks(ticks3, nums, fontsize=8)
    plt.xticks(ticks2, nums2, fontsize = 8) #use this line and the kine below for reduce parameters
    plt.yticks(ticks2, nums2, fontsize = 8)
    ax.tick_params(which = 'both', labeltop=False, labelbottom=True, top = False)
    #plt.xticks(fontsize=8)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # here you set the path and name you want the figure will be saved.
    plt.savefig(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\rat L5\trial\\' + 'r colormap reduce figure' + '.pdf',dpi=200)



    # plot relevant correlations p values

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    #you will be asked to choose a number for relevant correlations colormpa
    inp = input('choose 1 for original p\q colormap or 2 for 2 colors p\q colormap or 3 for 2 colors r colormap:')
    if inp == '2':
        cmap = colors.ListedColormap(['blue', 'white'])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        cax2 = ax2.matshow(p_fdr, vmin=0, vmax=1, cmap=cmap, picker=True, norm=norm) # for original p_value, insert correlations_p instead of p_fdr
        cbar = fig2.colorbar(cax2, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Significant', 'Not Significant'])
    elif inp == '3':
        cmap = colors.ListedColormap(['blue', 'white', 'red'])
        bounds = [-1, -0.0000001, 0.0000001, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        cax2 = ax2.matshow(p_fdr, vmin=-1, vmax=1, cmap=cmap, picker=True, norm=norm) # for original p_value, insert correlations_p instead of p_fdr
        cbar = fig2.colorbar(cax2, ticks=[-0.75, 0, 0.75])
        cbar.ax.set_yticklabels(['Negative', 'Irrelevant', 'positive'])

    else:
        #cax2 = ax2.matshow(p_fdr, vmin=-1, vmax=1, cmap=cm.coolwarm, picker=True)
        cax2 = ax2.matshow(p_fdr, norm=colors.LogNorm(vmin=0.1**6, vmax=0.05**1), cmap=cm.coolwarm_r, picker=True) # for original p_value, insert correlations_p instead of p_fdr
        fig2.colorbar(cax2)
    ax2.set_xticks(ticks, minor = True)
    ax2.set_yticks(ticks, minor = True)
    # plt.xticks(ticks3, nums, fontsize=8) #use this line and the kine below for all parameters
    # plt.yticks(ticks3, nums, fontsize=8)
    plt.xticks(ticks2, nums2, fontsize = 8)
    plt.yticks(ticks2, nums2, fontsize = 8)
    ax2.tick_params(which = 'both', labeltop=False, labelbottom=True, top = False)

    cid2 = fig2.canvas.mpl_connect('button_press_event', onclick)
    #here you set the path and name you want the figure will be saved.
    plt.savefig(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\\' + 'p color map blue figure' + '.pdf', dpi=200)

    #plt.show()



import pandas as pd
import hncUtility as hnc
import numpy as np
from matplotlib import pyplot as plt, cm, colors

if __name__ == '__main__':

#here you put the path mse data (with o values between MSEs) from Nitza
    df = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\rat L5\trial\mse data ratL5.csv')
#here you put the path dataset matrix of the dependent dataset (not our data)
    dft = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\rat L5\trial\ratL5 reduce.csv')



    if hasattr(dft, 'neuron_name'):
        print('removed neuron_name')
        dft.drop('neuron_name', inplace=True, axis=1)



    cols = dft.columns[:]
    p_mse_map = pd.DataFrame(index=cols, columns=cols, dtype='float')
    for ind in cols:
        for col in cols:
            p_mse_map.at[str(ind), str(col)] = 0.5

    Field_A = list(df['Field_A'])
    iter = range(len(Field_A))
    for it in iter:
        val = df['P-Value'][it] #in some mse data from Nitza, there is a coloumn 'p-value' (and not 'P-value') and you should change it respectively
        if val>=0.01: # here you set the threshold which p-values above him account not significant and hence similarity between the MSE.
            p_mse_map.at[df['Field_A'][it], df['Field_B'][it]] = 0
            p_mse_map.at[df['Field_B'][it], df['Field_A'][it]] = 0
        else:
            p_mse_map.at[df['Field_A'][it], df['Field_B'][it]] = 1
            p_mse_map.at[df['Field_B'][it], df['Field_A'][it]] = 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #cax = ax.matshow(p_mse_map, vmin=0.0005, vmax=0.1, cmap=cm.coolwarm_r, picker=True)
    cmap = colors.ListedColormap(['red', 'white', 'blue'])
    bounds = [0, 0.5, 1, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cax = ax.matshow(p_mse_map, vmin=0, vmax=1, cmap=cmap, picker=True, norm=norm)
    cbar=fig.colorbar(cax, ticks=[0.25,0.75,1.25])
    cbar.ax.set_yticklabels(['Significant', 'Irrelevant', 'Not Significant'])
    ticks = np.arange(0, len(cols), 1)
    ticks2 = [0, 4, 9, 14, 19, 24, 29, 34]
    nums = [x for x in range(1, 48) if x not in [1, 2, 3, 4, 5, 6, 7, 13, 20, 26, 41]] #numbers for reduced parameters.
    nums2 = [nums[i] for i in ticks2]
    # nums = np.array(indices)+1
    ax.set_xticks(ticks, minor = True)
    ax.set_yticks(ticks, minor = True)
    #ax.set_xticklabels(nums, {'fontsize': 8})
    #ax.set_yticklabels(nums, {'fontsize': 8})
    plt.xticks(ticks2, nums2, fontsize=8)
    plt.yticks(ticks2, nums2, fontsize=8)
    ax.tick_params(which = 'both', labeltop=False, labelbottom=True, top = False)
# write below the path and name you want to figure will be saved.
    plt.savefig(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\neuron_nmo\rat L5\trial\\' + 'p mse map figure' + '.pdf',
            dpi=200)

    # make a color map of fixed colors
    cmap = colors.ListedColormap(['white', 'red', 'blue'])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    #img = plt.imshow(zvals, interpolation='nearest', origin='lower',
    #                 cmap=cmap, norm=norm)

    # make a color bar
    #plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 5, 10])


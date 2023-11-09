import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
from matplotlib.colorbar import ColorbarBase
import numpy
import pandas as pd
from sklearn import preprocessing
import hncUtility as hnc
import pylab



if __name__ == '__main__':

    # put here the matrix path
    df = pd.read_csv(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\merged_alldatset! reduce.csv')

    if hasattr(df, 'neuron_name'):
        print('removed neuron_name')
        df.drop('neuron_name', inplace=True, axis=1)

    csv_field_names = df.columns
    colors = numpy.random.random((len(df.index), 3))
    df['colour'] = range(0, len(df.index), 1)
    pd.set_option('display.width', 1000)
    pd.set_option('precision',2)

    fig1 = plt.figure(figsize=(12,15), facecolor='white')
    ax1 = plt.axes(frameon=False)
    ax1.get_xaxis().tick_bottom()  # Turn off ticks at top of plot
    ax1.axes.get_yaxis().set_visible(False)  # Hide y axis
    ax1.set_xticks([-2, -1, 0, 1, 2])
    ax1.set_xticklabels(['-2', '-1', 'mean', '1', '2'], {'fontsize': 10})
    pylab.xlim([-3, 7])
    pylab.ylim([-1,len(csv_field_names)*0.55])
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    y = [18] # write 18 for reduced parameter distribution or 29.5 for all parameter distribution
    num = 1
    num = [x for x in range(1,48) if x not in [1,2,3,4,5,6,7,13,20,26,41]] #for reduced parameter distribution
    #num = [x for x in range(1,60)] #for all parameter distribution
    p = 'Parameters List:'
    df2 = df[:]
    df2, ol =hnc.remove_outliers(df2)
    j = 0
    for parameter, n in zip(csv_field_names, num):
        stand = df2[str(parameter)].std()
        mean_par = numpy.mean(df2[str(parameter)])
        df3 = df
        df3 = df3[df3[parameter].notnull()]
        if parameter == 'AD':
            show = 'n(0) = ' + str(sum(df2[parameter] == 0)) + ' n(1) = ' + str(
                sum(df2[parameter] == 1)) + ' n(2) = ' + str(sum(df2[parameter] == 2)) + ' n(3) = ' + str(
                sum(df2[parameter] == 3)) + ' ol = ' + str(ol[j])
        else:
            show = str(round(mean_par, 2)) + ' $ \pm $ ' + str(round(stand, 2)) + ', ' + \
                   str(sum(~numpy.isnan(df2[parameter]))) + ', ' + str(ol[j])
        scale = preprocessing.StandardScaler()
        if len(df3[parameter]) > 0:
            df3[parameter] = scale.fit_transform(df3[parameter].as_matrix())
        col_map = numpy.reshape(range(len(df3[parameter])*3), (len(df3[parameter]),3)).astype('float')
        ind = 0
        for i in df3['colour']:
            col_map[ind] = colors[i]
            ind += 1
        x = df3[parameter]
        if len(x) > 0:
            y = [y[0]] * len(x)
            ax1.scatter(x, y, s=50, facecolors='none', edgecolor=col_map)
        else:
            y = [y[0]]
        j += 1
        y = [y[0]-0.5]
        plt.text(4.2, 19, 'mean $ \pm $ std,  n,  outliers')
        plt.text(4.2, y[0]+0.5, show, fontsize=10)
        plt.text(-3.3, y[0]+0.5, n, fontsize=10)
        p += ('\n' + str(n) + '. ' + parameter.replace('_', ' '))
        #num += 1
    print(p)
    ax1.add_artist(Line2D((xmin, 4), (ymin,ymin), color='black', linewidth=3))
    #cmap = colors.ListedColormap(colors, N=None)
    #cbar = ColorbarBase(ax1,cmap=cmap, values=df['colour'], orientation='horizontal', ticks =df['colour'] )
    # write below the path and name for the figure you awnt to save
    plt.savefig(r'C:\Users\Rina\Studies\Msc\Research Dr Lior\my research\SWC datasets files\all dataset\\' + 'parameters distribution All2 reduce' + '.pdf', dpi = 200)
    plt.show()
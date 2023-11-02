#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
import numpy as np
import pandas as pd

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch

from mpl_toolkits.basemap import Basemap

from matplotlib import cm
from matplotlib import colors
from matplotlib.colorbar import ColorbarBase


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
#%%        
def agg_knr(df,func='mean'):
    """
    Function to aggregate df by municipality (knr)
    """
    if func=='mean':
        aggKnr = df.groupby('knr').mean()
    elif func=='sum':
        aggKnr = df.groupby('knr').sum()
    else:
        print('Func needs to be specified')
        aggKnr = None
    return aggKnr


def aggKnr_to_color(agg,colormapname='Reds'):
    """
    Normalize data for each knr and map into colormap
    """    
    #
    norm = matplotlib.colors.Normalize(vmin=np.min(agg),vmax=np.max(agg))

    #norm(0.)
    #
    cc = norm(agg.values)
    #
    rgb = cm.get_cmap(plt.get_cmap(colormapname))(cc)
    rgb = rgb[:,0:3]
    
    knrColor = pd.DataFrame(rgb,index=agg.index)
    
    return knrColor,norm

def plotMap(knrColor,figsize=(10, 10)):
    """
    Plot a base map using a shape file for Norwegian municipalities (knr) where
    each knr has a different color
    """
    fig = plt.figure(figsize=figsize)
    map = Basemap(llcrnrlon=4.086914,llcrnrlat=57.685358,urcrnrlon=33.442383,urcrnrlat=71.289518,
                 resolution='l', projection='tmerc', lat_0 = 65.094346, lon_0 = 11.381836)

    #map.drawmapboundary(fill_color='aqua')
    #map.fillcontinents(color='#ddaa66',lake_color='aqua')
    #map.drawcoastlines()

    map.readshapefile('Z:/data/shapefiles/NOR2Full', 'NOR2Full')

    ax = plt.gca() 

    for info, shape in zip(map.NOR2Full_info, map.NOR2Full):
        if info['NO_ID'] in knrColor.index:
            patches   = []
            patches.append( Polygon(np.array(shape), True) )
            ax.add_collection(PatchCollection(patches, facecolor=list(knrColor.loc[info['NO_ID']]), edgecolor='k', linewidths=1., zorder=2))

    ax.axis('off')
    
   
    return fig,plt,ax
#%%
def plot_agg_map(df,selVar,func='mean',colormapname='Reds',figsize=(10, 10),saveName=None):
    """
    Wrapper function using the previous functions to
    aggregate data and plot a map
    """
    #%% Aggregate data by knr
    aggKnr = agg_knr(df,func)
    #%% Map aggregate values to color bar range (-1,1)
    knrColor,norm = aggKnr_to_color(aggKnr[selVar],colormapname)
    #%% Plot map
    fig, plt, ax = plotMap(knrColor,figsize=figsize)
    
    #%% construct custom colorbar
    cmap = plt.get_cmap(colormapname)
    cax = fig.add_axes([0.4, 0.1, 0.3, 0.03]) # posititon
    cb = ColorbarBase(cax,cmap=cmap,norm=norm, orientation='horizontal')
    cb.ax.set_xlabel(func+' of '+selVar+' per Municipality')
    
    plt.show()
    
    if saveName!=None:
        fig.savefig(saveName+'_'+selVar+'_'+func+'.png')
    return fig, plt, ax, aggKnr
        
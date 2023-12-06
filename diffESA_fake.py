"""plots maps iof CCClim and the comparison to syntehtci ICON observations

"""
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

ctnames = ["Ci","As","Ac","St","Sc", "Cu", "Ns","Dc"]
work = os.environ["WORK"]
scratch = os.environ["SCR"]

def customize(ax,left=True,bottom=True):
    """
    
    makes a good looking world map plot
    Parameters
    ----------
    ax : matplotlib axis
        axis to customize

    Returns
    -------
    None.

    """
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels={"top":False,"bottom":bottom, "left":left,
                    "right":False},y_inline=False)
    ax.tick_params(labelsize=16, )
    ax.coastlines()


if __name__=="__main__":
    #cloud type distributions from synthetic icon observations
    ICON = np.load(os.path.join(os.environ["HOME"],"ML/fakeICON_gentarget2sourcecycle_96_5_full_run_800best.npz"))
    #the CCClim data in a usable dataframe, as 1° grid box averages, but timeresolved.
    #this one contains 967 days
    ESA = pd.read_pickle(os.path.join(os.environ["HOME"],"ML/dfglobe.pkl"))
    #turns the multiindex (lat,lon) into a hierarchical index
    #lat now indexes the rows, lon and the properties index the columns
    ESA =ESA.unstack()
    
    cmax = max([np.max(ICON[x][0]) for x in ctnames ])
    #for colorbar scaling or something like that
    cmax = max(cmax,np.max(ESA[ctnames].values))
    print(cmax)
    fig,ax = plt.subplots(4,4,figsize=(16,9),subplot_kw = {"projection":ccrs.Robinson()},
                            gridspec_kw={"wspace":0.1,"hspace":0.05},
                            sharex=True,sharey=True)
    
    for i_tot,cname in enumerate(ctnames):
        ICON_ct,lonlon,latlat = ICON[cname]
        ESA_ct = ESA[cname].values[:-1,:-1]
        i=i_tot%4 # for finding the right axis to plot in
        s= int(i_tot>3)
        print(i,i_tot,s)
        # the rest here is straightforward plotting
        print(np.max(np.abs(ESA_ct-ICON_ct)))
        if i==0:
            labloc=-0.15
        else:
            labloc=-0.05        
        
        abs=ax[0+s,i].pcolormesh(lonlon,latlat,ICON_ct,vmin=0,vmax=cmax,cmap="gist_stern",
                            transform=ccrs.PlateCarree(),rasterized=True)
        ax[0+s,i].text(labloc,0.5,cname,horizontalalignment='center', verticalalignment='center',
                        transform=ax[0+s,i].transAxes,fontsize=18 )
        diff=ax[2+s,i].pcolormesh(lonlon,latlat,ESA_ct-ICON_ct,vmin=-0.6,vmax=0.6,cmap="seismic",
                            transform=ccrs.PlateCarree(),rasterized=True)
        ax[2+s,i].text(labloc,0.5,cname,horizontalalignment='center', verticalalignment='center',
                        transform=ax[2+s,i].transAxes,fontsize=18 )
        for j in range(4):
            customize(ax[j,i],left=i_tot==0,bottom=j==3)
    iconlabel = fig.add_axes((0.004,.728, 0.015,0.04))
    iconlabel.set_axis_off()
    iconlabel.text(0,0,"Synth. Obs. of ICON-A",rotation="horizontal",fontsize=20)
    difflabel = fig.add_axes((0.004,.23, 0.015,0.0))
    difflabel.set_axis_off()
    difflabel.text(0,0,"Diff. to\nCCClim",rotation="horizontal",fontsize=20)
    fig.subplots_adjust(left=0.05,bottom=0.01,top=0.98,right=0.95)
    diffbar = fig.add_axes((0.951,.06, 0.02,0.4))
    fig.colorbar(diff,cax = diffbar,orientation="vertical")
    absbar = fig.add_axes((0.951,.55, 0.02,0.4))
    fig.colorbar(abs,cax = absbar,orientation="vertical")
    fig.savefig(os.path.join(work,"stats/diffESA_fake.pdf"))
"""TLLT plots"""

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr            # pip install cmasher
import cartopy as ct
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy.ma as ma
import copy


__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "20 August 2021"

def plot_mask(ax,image):
    p = ax.pcolor(image,
                  cmap = 'Greys'
#                   cmap='cmr.flamingo_r',                  
#                   cmap = 'cmr.bubblegum_r'
#                   cmap = 'cmr.ocean_r'                  
                 )
    ax.set_xticks([])
    ax.set_yticks([])
    
    return p

def plot_sample(ax,image,globe=False, lat=None, lon=None, mapProj=None):
    
    if globe:
#         mapProj = ct.crs.EqualEarth(central_longitude = 180.)
        cb, p = drawOnGlobe(ax, 
                                mapProj,
                                image, 
                                lat, 
                                lon, 
                                cbarBool=False, 
                                fastBool=True,
                                cmap='cmr.fusion_r',
                                vmin=-5,
                                vmax=5,
                               )
        p.set_clim(-5,5)
        
        ax.set_extent([np.min(lon), 
                       np.max(lon), 
                       np.min(lat), 
                       np.max(lat)], 
                      crs=ccrs.PlateCarree())        
        
    else:
        p = ax.contourf(image,
#                       levels = np.arange(-10., 10.5, .5), 
                      levels = np.arange(-10., 10.25, .25),                         
                      cmap='cmr.fusion_r',                  
                     )
        p.set_clim(-10,10)
#         p.set_clim(-8,8)
        
    ax.set_xticks([])
    ax.set_yticks([])
    
    return p

def plot_sample_shaded(ax,image,globe=False, lat=None, lon=None, mapProj=None, rf=None):
    
    if globe:
        # plot the sample
        p = plot_sample(ax,image,globe=True,lat=lat,lon=lon, mapProj=mapProj)
        
        # shade out non-prototype areas.
        q = ax.pcolor(lon,
                       lat,
                       rf,
                       cmap='Greys',
                       alpha=.6,
                       transform=ccrs.PlateCarree(),
                      )        
        q.set_clim(0,3)
        
    else:
        # plot the sample
        p = plot_sample(ax,image,globe=False)
        
        # shade out non-prototype areas.
        q = ax.pcolor(rf,
                       cmap='Greys',
                       alpha=.6,
                      )        
        q.set_clim(0,3)
        
        
    ax.set_xticks([])
    ax.set_yticks([])
    
    return p

def display_rfpatch(ax, image, clr='gray', globe=False, lat=None, lon=None):
    
    if globe:
        p = ax.contour(lon,lat,image, 
                      colors=clr, 
                      alpha = 0.5,
                      linewidths=.5,
                      transform=ccrs.PlateCarree()
                  )        
    else:
        p = ax.contour(image, 
                      colors=clr, 
                      alpha = 0.5,
                      linewidths=.5,
                  )
    return p


def plot_weights(model, prototypes_per_class=None):
    w           = model.layers[-2].get_weights()[0]
    nprototypes = w.shape[0]
    nclasses    = w.shape[1]
    
    if(prototypes_per_class is None):
        prototypes_per_class = np.ones(shape=(nclasses,))*int(nprototypes/nclasses)
    else:
        assert(nprototypes == np.sum(prototypes_per_class))
    
    
    plt.figure(figsize=(9,5))
    plt.axhline(y=0,color='gray',linewidth=.5)

    for c in range(nclasses):
        p = plt.plot(np.arange(nprototypes)+.05*c,
                     w[:,c], 
                     '.--',
                     label = 'class ' + str(c),
                     linewidth=2,
                     markersize=12,
                    )
        
        color           = p[0].get_color()
        left_prototype  = np.sum(prototypes_per_class[:c])
        right_prototype = np.sum(prototypes_per_class[:c+1])
        plt.axvspan(left_prototype-.5, right_prototype-.5, alpha=0.15, color=color)

    plt.legend(fontsize=8, loc=9, ncol=w.shape[1])    

    plt.ylim(-.75,2.)
    plt.ylabel('weight')

    plt.xticks(range(0,nprototypes+1))
    plt.xlim(-.5,nprototypes-.5)
    plt.xlabel('prototype number')

    plt.title('Weights Connected to Output Classes')

    plt.tight_layout()
    
def plot_loss_history(history):
    
    trainColor = (117/255., 112/255., 179/255., 1.)
    valColor = (231/255., 41/255., 138/255., 1.)
    FS = 7
    MS = 4

    plt.subplots(1,2,figsize=(10, 3))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], 'o-', color=trainColor, label='training', markersize=MS)
    plt.plot(history.history['val_loss'], 'o-', color=valColor, label='validation', markersize=MS)
    # plt.axvline(x=best_epoch, linestyle = '--', color='tab:gray')
    plt.title("Loss Function")
    plt.ylabel('average loss')
    plt.xlabel('epoch')
    plt.grid(False)
    plt.legend(frameon=True, fontsize=FS)
    plt.xlim(-.1, len(history.history['loss'])+1)

    plt.subplot(1,2,2)
    plt.plot(history.history['sparse_categorical_accuracy'], 'o-', color=trainColor, label='training', markersize=MS)
    plt.plot(history.history['val_sparse_categorical_accuracy'], 'o-', color=valColor, label='validation', markersize=MS)
    # plt.axvline(x=best_epoch, linestyle = '--', color='tab:gray')
    plt.title("Accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(False)
    plt.legend(frameon=True, fontsize=FS)
    plt.xlim(-.1, len(history.history['loss'])+1)

    plt.tight_layout()

    
def drawOnGlobe(ax, map_proj, data, lats, lons, cmap='coolwarm', vmin=None, vmax=None, inc=None, cbarBool=True, contourMap=[], contourVals = [], fastBool=False, extent='both'):

    data_crs = ct.crs.PlateCarree()
    data_cyc, lons_cyc = add_cyclic_point(data, coord=lons) #fixes white line by adding point#data,lons#ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point
    data_cyc = data
    lons_cyc = lons
    
    
#     ax.set_global()
#     ax.coastlines(linewidth = 1.2, color='black')
#     ax.add_feature(cartopy.feature.LAND, zorder=0, scale = '50m', edgecolor='black', facecolor='black')    
    land_feature = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        facecolor='None',
        edgecolor = 'k'
    )
    ax.add_feature(land_feature)
#     ax.GeoAxes.patch.set_facecolor('black')
    
    if(fastBool):
        image = ax.pcolormesh(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap)
#         image = ax.contourf(lons_cyc, lats, data_cyc, np.linspace(0,vmax,20),transform=data_crs, cmap=cmap)
    else:
        image = ax.pcolor(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap,shading='auto')
    
    if(np.size(contourMap) !=0 ):
        contourMap_cyc, __ = add_cyclic_point(contourMap, coord=lons) #fixes white line by adding point
        ax.contour(lons_cyc,lats,contourMap_cyc,contourVals, transform=data_crs, colors='fuchsia')
    
    if(cbarBool):
        cb = plt.colorbar(image, shrink=.5, orientation="horizontal", pad=.02, extend=extent)
        cb.ax.tick_params(labelsize=6) 
    else:
        cb = None

    image.set_clim(vmin,vmax)
    
    return cb, image   

def add_cyclic_point(data, coord=None, axis=-1):

    # had issues with cartopy finding utils so copied for myself
    
    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError('The length of the coordinate does not match '
                             'the size of the corresponding dimension of '
                             'the data array: len(coord) = {}, '
                             'data.shape[{}] = {}.'.format(
                                 len(coord), axis, data.shape[axis]))
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError('The coordinate must be equally spaced.')
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value
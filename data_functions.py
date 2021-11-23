import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "23 November 2021"

def makeRasterArray(fig, ax, fig_shape=(4,3)):

    fig.add_axes(ax)
    fig.set_dpi(100)
    ax.axis('off')
    fig.set_size_inches(fig_shape)  # fig_shape in inches
    
    return makeRasterArray_main(fig)

def makeRasterArray_main(fig):
    """ Make a raster array from a pyplot figure.

    References:
       https://matplotlib.org/stable/gallery/misc/agg_buffer_to_array.html
       https://stackoverflow.com/questions/51059581/matplotlib-convert-plot-to-numpy-array-without-borders

    """
    
    fig.canvas.draw()                                      # Force a draw so we can grab the pixel buffer.
    return np.array(fig.canvas.renderer.buffer_rgba())     # Grab the pixel buffer and dump it into a numpy array.


def get_and_process_mjo_data(raw_labels, raw_data, raw_time, rng, colored=False, standardize=False, shuffle=True):
    
    if(colored == True):
        raw_data = np.abs(1. - np.asarray(raw_data, dtype='float')/255.)
    else:
        raw_data = np.asarray(raw_data, dtype='float')
    raw_labels   = np.asarray(raw_labels, dtype='float')

    if(shuffle==True):
        # shuffle the data
        print('shuffling the data before train/validation/test split.')
        index      = np.arange(0,raw_data.shape[0])
        rng.shuffle(index)  
        raw_data    = raw_data[index,:,:,:]
        raw_labels  = raw_labels[index,]

    # separate the data into training, validation and testing
    all_years = raw_time["time.year"].values
    years     = np.unique(all_years)

    nyr_val   = int(len(years)*.2)    
    nyr_train = len(years) - nyr_val
    
    years_train = np.random.choice(years,size=(nyr_train,),replace=False)
#     years_train = rng.choice(years,size=(nyr_train,),replace=False)     #use this syntax next time to keep everything using rng
    years_val   = np.setxor1d(years,years_train)
    years_test  = 2010
    
    ic(years_train)
    ic(years_val)
    ic(years_test)    
    
    years_train = np.isin(all_years, years_train)
    years_val   = np.isin(all_years, years_val)
    years_test  = np.isin(all_years, years_test)
    
    iyears_train = np.where(years_train==True)[0]
    iyears_val = np.where(years_val==True)[0]
    iyears_test = np.where(years_test==True)[0]   
        
    # Standardize the input based on training data only
    X_train_raw = raw_data[iyears_train,:,:,:]
    if( (standardize==True) or (standardize=='all')):
        X_mean  = np.mean(X_train_raw.flatten())
        X_std   = np.std(X_train_raw.flatten())
    elif(standardize=='pixel'):
        X_mean  = np.mean(X_train_raw,axis=(0,))
        X_std   = np.std(X_train_raw,axis=(0,))
        X_std[X_std==0] = 1.
    else:
        X_mean  = 0. 
        X_std   = 1. 
    
    # Create the target vectors, which includes a second dummy column.
    X_train = (raw_data[iyears_train,:,:,:] - X_mean) / X_std
    X_val   = (raw_data[iyears_val,:,:,:] - X_mean) / X_std
    X_test  = (raw_data[iyears_test,:,:,:] - X_mean) / X_std

    X_train[X_train==0.] = 0.
    X_val[X_val==0.]     = 0.
    X_test[X_test==0.]   = 0.    
    
    y_train = raw_labels[iyears_train]
    y_val   = raw_labels[iyears_val]
    y_test  = raw_labels[iyears_test]

    time_train = raw_time[iyears_train]
    time_val   = raw_time[iyears_val]
    time_test  = raw_time[iyears_test]
    
    
    print(f"raw_data        = {np.shape(raw_data)}")
    print(f"training data   = {np.shape(X_train)}, {np.shape(y_train)}")
    print(f"validation data = {np.shape(X_val)}, {np.shape(y_val)}")
    print(f"test data       = {np.shape(X_test)}, {np.shape(y_test)}")
    if(standardize != 'pixel'):
        print(f"X_mean          = {X_mean}")
        print(f"X_std           = {X_std}")    
    else:
        print(f"X_mean.shape    = {X_mean.shape}")
        print(f"X_std.shape     = {X_std.shape}")    
        
    
    return (X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test)

def cut_longitudes(data,lon):
    ilon = np.where(np.logical_and(lon<=270,lon>=10))[0] #240
#     ilon = np.where((lon.values<=360) & (lon.values>=0))[0]
    lon = lon[ilon]
    data = data[:,:,ilon]    
    return data, lon

def load_mjo_data(load_dir):
    
    # make labels
    filename = 'WH04_RMM.nc'
    time     = xr.open_dataset(load_dir+filename)['RMM_amp']['time']    
    rmm_amp  = xr.open_dataset(load_dir+filename)['RMM_amp']
    rmm_ph   = xr.open_dataset(load_dir+filename)['RMM_ph']
    
    i = np.where(rmm_amp<.5)
    rmm_ph[i] = 0

    print(np.unique(rmm_ph))    
    
    # get the fields
    filename = 'olr.20NS.noac_120mean_norm.nc'
    lat      = xr.open_dataset(load_dir+filename)['olr']['latitude']
    lon      = xr.open_dataset(load_dir+filename)['olr']['longitude']
    olr      = xr.open_dataset(load_dir+filename)['olr'].values[:,:,:,np.newaxis]
    
    filename = 'u200.20NS.noac_120mean_norm.nc'
    u200     = xr.open_dataset(load_dir+filename)['u'].values[:,:,:,np.newaxis] 
    
    filename = 'u850.20NS.noac_120mean_norm.nc'
    u850     = xr.open_dataset(load_dir+filename)['u'].values[:,:,:,np.newaxis]
    
    olr, __   = cut_longitudes(olr,lon)
    u200, __  = cut_longitudes(u200,lon)
    u850, lon = cut_longitudes(u850,lon)
    data = np.concatenate((olr,u200,u850),axis=3)
    
    return rmm_ph, data, lat, lon, time


def get_and_process_data(filename, rng, colored=False, standardize=False, shuffle=True):
    print('loading ' + filename)
    
    mat_contents = sio.loadmat(filename)
    raw_data     = mat_contents['data']
    raw_labels   = mat_contents['labels'].T[:,0]
    try:
        lat      = mat_contents['lat'][0]
        lon      = mat_contents['lon'][0]
    except:
        lat      = np.nan
        lon      = np.nan
        
    if(colored == True):
        raw_data = np.abs(1. - np.asarray(raw_data, dtype='float')/255.)
    else:
        raw_data = np.asarray(raw_data, dtype='float')
    raw_labels   = np.asarray(raw_labels, dtype='float')

    if(shuffle==True):
        # shuffle the data
        print('shuffling the data before train/validation/test split.')
        index      = np.arange(0,raw_data.shape[0])
        rng.shuffle(index)  
        raw_data    = raw_data[index,:,:,:]
        raw_labels  = raw_labels[index,]

    # separate the data into training, validation and testing
    ndata       = len(raw_data)
    ntest       = np.int32(0.000001 * ndata)
    nval        = np.int32(0.2 * ndata)
#     ntest       = np.int32(0.001 * ndata)
#     nval        = np.int32(0.25 * ndata)
    ntrain      = np.int32(ndata - nval - ntest)

    # Standardize the input based on training data only
    X_train_raw = raw_data[0:ntrain,:,:]
    if( (standardize==True) or (standardize=='all')):
        X_mean  = np.mean(X_train_raw.flatten())
        X_std   = np.std(X_train_raw.flatten())
    elif(standardize=='pixel'):
        X_mean  = np.mean(X_train_raw,axis=(0,))
        X_std   = np.std(X_train_raw,axis=(0,))
        X_std[X_std==0] = 1.
    else:
        X_mean  = 0. 
        X_std   = 1. 

    X_train = (raw_data[0:ntrain,:,:] - X_mean) / X_std
    X_val   = (raw_data[ntrain:ntrain+nval,:,:] - X_mean) / X_std
    X_test  = (raw_data[ntrain+nval:,:,:] - X_mean) / X_std

    y_train = raw_labels[0:ntrain]
    y_val   = raw_labels[ntrain:ntrain+nval]
    y_test  = raw_labels[ntrain+nval:]

    if(len(np.shape(X_train))==3):
        X_train = np.asarray(X_train)[:,:,:,np.newaxis]
        X_val   = np.asarray(X_val)[:,:,:,np.newaxis]    
        X_test  = np.asarray(X_test)[:,:,:,np.newaxis]        
    
    print(f"raw_data        = {np.shape(raw_data)}")
    print(f"training data   = {np.shape(X_train)}, {np.shape(y_train)}")
    print(f"validation data = {np.shape(X_val)}, {np.shape(y_val)}")
    print(f"test data       = {np.shape(X_test)}, {np.shape(y_test)}")
    if(standardize != 'pixel'):
        print(f"X_mean          = {X_mean}")
        print(f"X_std           = {X_std}")    
    else:
        print(f"X_mean.shape    = {X_mean.shape}")
        print(f"X_std.shape     = {X_std.shape}")    
   
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, lat, lon)

def subsample_extremes(random_seed, X_train, y_train, X_val, y_val, X_test, y_test):
    # subsample training data
    np.random.seed(random_seed)
    labels0 = np.where(y_train==0)[0]
    labels1 = np.where(y_train==1)[0]
    labels2 = np.where(y_train==2)[0]    
    labels0_subsample = np.random.choice(labels0,size=len(labels1),replace=False)
    X_train = X_train[np.concatenate((labels0_subsample,labels1,labels2)),:,:,:]
    y_train = y_train[np.concatenate((labels0_subsample,labels1,labels2))]

    # subsample validation data
    np.random.seed(random_seed)
    labels0 = np.where(y_val==0)[0]
    labels1 = np.where(y_val==1)[0]
    labels2 = np.where(y_val==2)[0]    
    labels0_subsample = np.random.choice(labels0,size=len(labels1),replace=False)
    X_val = X_val[np.concatenate((labels0_subsample,labels1,labels2)),:,:,:]
    y_val = y_val[np.concatenate((labels0_subsample,labels1,labels2))]

    # show it worked
    labels0 = np.where(y_train==0)[0]
    labels1 = np.where(y_train==1)[0]
    labels2 = np.where(y_train==2)[0]    
    ic(len(labels0),len(labels1),len(labels2))

    labels0 = np.where(y_val==0)[0]
    labels1 = np.where(y_val==1)[0]
    labels2 = np.where(y_val==2)[0]    
    __ = ic(len(labels0),len(labels1),len(labels2))
    
    return X_train, y_train, X_val, y_val, X_test, y_test
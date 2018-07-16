import sys
sys.path.append('../libs/')

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters
from Quaternions import Quaternions
from Pivots import Pivots
import glob


import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.animation import ArtistAnimation
import matplotlib.patheffects as pe



def preprocess(data, window=256, window_step=128):
    original_positions = np.array(data).T.reshape(-1, 32, 3)#[0:10]
    positions = original_positions[:, np.array([
        #0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        #11,#
        12,
        13,
        14,
        15,
        #16,#
        17,
        18,
        #19,#
        20,
        21,
        22,
        #23,#
        #24,#
        25,
        26,
        #27,#
        28,
        29,
        30,
        #31#
    ])]#[0:10]
    
    
    """ Add Reference Joint """
    #trajectory_filterwidth = 3
    #reference = original_positions[:,0] * np.array([1,1,1])
    #reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
    #positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
    
        
    """ Get Root Velocity """
    velocity = original_positions[1:,0:1] - original_positions[:-1,0:1]
    
    """ Remove Translation """
    positions[:,:] = positions[:,:] - original_positions[:,0:1]
    
    """ Get Forward Direction """
    spine, hip_l, hip_r = 10, 5, 0
    normal = np.cross(positions[:,hip_l] - positions[:,spine], positions[:,hip_r] - positions[:,hip_l])
    normal = normal / np.sqrt((normal**2).sum(axis=-1))[...,np.newaxis]
    
    """ Remove Z Rotation """
    lever = np.cross(normal, np.array([[0,0,1]])) 
    lever = lever / np.sqrt((lever**2).sum(axis=-1))[...,np.newaxis]
    target = np.array([[1,0,0]]).repeat(len(lever), axis=0)
    rotation = Quaternions.between(lever, target)[:,np.newaxis]    
    positions = rotation * positions
    
    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    #rvelocity = (rotation[1:] * -rotation[:-1]).euler()
    
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1], forward="y", plane="xy").ps
    
    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, np.ones(shape=(len(positions), 1))], axis=-1)
    positions = np.concatenate([positions, velocity.reshape(-1, 3)], axis=-1)
    positions = np.concatenate([positions, rvelocity.reshape(-1, 1)], axis=-1)

    """ Slide over windows """
    windows = []
    
    for j in range(0, len(positions)-window//8, window_step):
    
        """ If slice too small pad out by repeating start and end poses """
        slice = positions[j:j+window]
        if len(slice) < window:
            left  = np.zeros(shape=slice[:1].shape).repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
            right = np.zeros(shape=slice[:1].shape).repeat((window-len(slice))//2, axis=0)
            slice = np.concatenate([left, slice, right], axis=0)
        
        if len(slice) != window: raise Exception()
        
        windows.append(slice)
    
    return windows


trainfiles = glob.glob("../data/raw/h36m/S[15678]/MyPoses/3D_positions/*.h5")
windows = []
for file in trainfiles:
    windows += preprocess(h5py.File(file)["3D_positions"])
Xtrain = np.array(windows)


validfiles = glob.glob("../data/raw/h36m/S9/MyPoses/3D_positions/*.h5")
validfiles.extend(glob.glob("../data/raw/h36m/S11/MyPoses/3D_positions/*.h5"))
windows = []
for file in validfiles:
    windows += preprocess(h5py.File(file)["3D_positions"])
Xvalid = np.array(windows)

Xmean = np.mean(Xtrain, axis=(0,1))
Xstd = np.std(Xtrain, axis=(0,1))

# Do that in preprocessing!
#Xtrain = (Xtrain - Xmean) / Xstd
#Xvalid = (Xvalid - Xmean) / Xstd

np.random.shuffle(Xtrain)
np.random.shuffle(Xvalid)

np.savez("../data/windows/h36m.npz",
    Xtrain=Xtrain.astype(np.float32),
    Xvalid=Xvalid.astype(np.float32)
)

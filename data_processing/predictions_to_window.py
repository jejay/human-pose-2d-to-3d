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
    original_positions = np.array(data).reshape(-1, 32, 3)#[0:10]
    positions = original_positions[:, np.array([
        #0, #Hips (Hip)
        1,  #RightUpLeg (RHip)
        2,  #RightLeg (RKnee)
        3,  #RightFoot (RFoot)
        0,#4,  #RightToeBase
        0,#5,  #Site
        6,  #LeftUpLeg (LHip)
        7,  #LeftLeg (LKnee)
        8,  #LeftFoot (LFoot)
        0,#9,  #LeftToeBase
        0,#10, #Site
        #11,#Spine
        12, #Spine1 (Spine)
        13, #Neck (Thorax)
        14, #Head (Neck/Nose)
        15, #Site (Head)
        #16,#LeftShoulder
        17, #LeftArm (LShoulder)
        18, #LeftForeArm (LElbow)
        #19,#LeftHand (LWrist)
        19,#20, #LeftHandThumb
        0,#21, #Site
        0,#22, #L_Wrist_End
        #23,#Site
        #24,#RightShoulder
        25, #RightArm (RShoulder)
        26, #RightForeArm (RElbow)
        #27,#RightHand (RWrist)
        27,#28, #RightHandThumb
        0,#29, #Site
        0,#30, #R_Wrist_End
        #31 #Site
    ])]#[0:10]
    
    """ Add Reference Joint """
    #trajectory_filterwidth = 3
    #reference = original_positions[:,0] * np.array([1,1,1])
    #reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
    #positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
    ### Stuff is missing here!!!!!
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, np.ones(shape=(len(positions), 1))], axis=-1)
    positions = np.concatenate([positions, np.zeros(shape=(len(positions), 4))], axis=-1)

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


trainfiles = glob.glob("../data/raw/predictions-walking.npz")
windows = []
for file in trainfiles:
    windows += preprocess(np.load(file)["poses3d"])
Xpredictions = np.array(windows)

np.savez("../data/windows/predictions-walking.npz",
    Xpredictions=Xpredictions.astype(np.float32),
)

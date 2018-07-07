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
    
    return positions
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


#files = glob.glob("h36m/S[15678]/MyPoses/3D_positions/*.h5")
#files = glob.glob("h36m/S9/MyPoses/3D_positions/*.h5")
#files.extend(glob.glob("h36m/S11/MyPoses/3D_positions/*.h5"))
#windows = []
#for file in files:
#    windows += preprocess(h5py.File(file)["3D_positions"])

#X = np.array(windows)

#X_mean = np.mean(X, axis=(0,1))
#X_std = np.std(X, axis=(0,1))

#X = (X - X_mean) / X_std
#np.random.shuffle(X)

#np.savez("h36m-validation.npz", X=X.astype(np.float32))#, X_mean=X_mean.astype(np.float32), X_std=X_std.astype(np.float32))




scale = 20
spine, hip_l, hip_r = 10, 5, 0
animations = [preprocess(h5py.File("h36m/S1/MyPoses/3D_positions/Walking.h5")["3D_positions"])]

for ai in range(len(animations)):
    if ai == 0:
        anim = animations[ai]
        
        joints, vel, rot = anim[:,0:-5], anim[:,-4:-1], anim[:,-1]
        joints = joints.reshape((len(joints), -1, 3))
        
        rotation = Quaternions.id(1)
        offsets = []
        translation = np.array([[0,0,0]])
        
        #if not ignore_root:
        for i in range(len(joints)):
            joints[i,:,:] = rotation * joints[i] + translation
            rotation = Quaternions.from_angle_axis(rot[i], np.array([0,0,1])) * rotation
            translation = translation + rotation * vel[i]
    
        animations[ai] = joints

interval = 20

scale = 30

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-scale*30, scale*30)
ax.set_zlim3d( 0, scale*60)
ax.set_ylim3d(-scale*30, scale*30)
ax.set_xticks([], [])
ax.set_yticks([], [])
ax.set_zticks([], [])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect('equal')

acolors = list(sorted(colors.cnames.keys()))[::-1]
lines = []

for ai, anim in enumerate(animations):
    lines.append([plt.plot([0], [0], [0], (["b" if _ == spine else "r" if _ == hip_l else "y" if _ == hip_r else "k","g"])[ai]+".", markersize=3)[0] for _ in range(anim.shape[1])])

def animate(i):
    changed = []
    
    for ai in range(len(animations)):
        
        for j in range(animations[ai].shape[1]):
            lines[ai][j].set_data(animations[ai][i,j,0], animations[ai][i,j,1])
            lines[ai][j].set_3d_properties(animations[ai][i,j,2])
        changed += lines[ai]
        
    return changed
    
plt.tight_layout()
    
ani = animation.FuncAnimation(fig, 
    animate, np.arange(len(animations[0])), interval=interval)


plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.animation import ArtistAnimation
import matplotlib.patheffects as pe

from Quaternions import Quaternions

predictions = np.load("predictions.npz")
manifolded = np.load("manifolded.npz")

preds = predictions["poses3d"].reshape(-1, 32, 3)[:,:, [0,2,1]]*0.0177167*0.9
gold = predictions["dec_out"].reshape(-1, 32, 3)[:,:, [0,2,1]]*0.0177167*0.9



import scipy.ndimage.filters as filters
from Pivots import Pivots

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))
def softmin(x, **kw):
    return -softmax(-x, **kw)

fid_l, fid_r = np.array([8]), np.array([3])
foot_heights = np.minimum(preds[:,fid_l,1], preds[:,fid_r,1]).min(axis=1)
floor_height = softmin(foot_heights, softness=0.5, axis=0)

preds[:,:,1] -= floor_height


""" Get Root Velocity """
velocity = (preds[1:,0:1] - preds[:-1,0:1]).copy()

#sdr_l, sdr_r, hip_l, hip_r = 17, 25, 6, 1
sdr_l, sdr_r, hip_l, hip_r = 25, 17, 1, 6
across1 = preds[:,hip_l] - preds[:,hip_r]
across0 = preds[:,sdr_l] - preds[:,sdr_r]
across = across0 + across1
across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

direction_filterwidth = 20
forward = np.cross(across, np.array([[0,1,0]]))
forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

""" Remove Y Rotation """
target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
rotation = Quaternions.between(forward, target)[:,np.newaxis]    
preds = rotation * preds    

""" Get Root Rotation """
velocity = rotation[1:] * velocity
rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps


preds_conv = np.concatenate(
    [
     np.zeros(shape=(preds.shape[0], 1 * 3)),
     preds[:,:4,:].reshape(-1, 4*3), #H36M_NAMES[0]  = 'Hip' H36M_NAMES[1]  = 'RHip' H36M_NAMES[2]  = 'RKnee' H36M_NAMES[3]  = 'RFoot'
     np.zeros(shape=(preds.shape[0], 1 * 3)), #RFootToe
     preds[:,6:9,:].reshape(-1, 3*3), #H36M_NAMES[6]  = 'LHip' H36M_NAMES[7]  = 'LKnee' H36M_NAMES[8]  = 'LFoot'
     np.zeros(shape=(preds.shape[0], 2 * 3)), #LFootToe, untere wirbelsäule
     preds[:,12:14,:].reshape(-1, 2*3), #H36M_NAMES[12] = 'Spine' H36M_NAMES[13] = 'Thorax
     preds[:,15,:].reshape(-1, 1*3), #H36M_NAMES[15] = 'Head'
     preds[:,25:28,:].reshape(-1, 3*3), #H36M_NAMES[25] = 'RShoulder' H36M_NAMES[26] = 'RElbow' H36M_NAMES[27] = 'RWrist'
     np.zeros(shape=(preds.shape[0], 1 * 3)), #RHand
     preds[:,17:20,:].reshape(-1, 3*3), #H36M_NAMES[17] = 'LShoulder' H36M_NAMES[18] = 'LElbow' H36M_NAMES[19] = 'LWrist'
     np.zeros(shape=(preds.shape[0], 1 * 3)), #LHand
     np.zeros(shape=(preds.shape[0], 2)),
     np.concatenate([rvelocity, [[0]]], axis=0),
     np.zeros(shape=(preds.shape[0], 4))
    ], axis=1).transpose().reshape(1, 73, -1)

    
    
    
    
    
import tensorflow as tf
from SingleManifoldModel import SingleManifoldModel

tf.reset_default_graph()
rng = np.random.RandomState(23455)

params = np.load("network_core.npz")
preprocess = np.load('preprocess_core.npz')
X = (preds_conv - preprocess['Xmean']) / preprocess['Xstd']

inputs = tf.placeholder(dtype=tf.float32, shape=[None, 73, X.shape[2]]) # (batchsize, 73, window)
model = SingleManifoldModel(window=X.shape[2])
model.build_graph(inputs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

index = 0 #rng.randint(X.shape[0])
Xorgi = np.array(X[index:index+1])
Xnois = np.array(X[index:index+1])
Xrecn = np.array(sess.run(model.autoencoded, {
    inputs: Xnois,
    'single_layer_manifold/conv1d/kernel:0': params['L000_L001_W'].transpose(),
    'single_layer_manifold/conv1d/bias:0': params['L000_L002_b'].reshape((256,)),
    'single_layer_manifold/conv1d_1/kernel:0': params['L001_L002_W'].transpose(),
    'single_layer_manifold/conv1d_1/bias:0': params['L001_L003_b'].reshape((73,)),
}))    

Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    
    
    
    
    
animations = [gold, Xnois, Xrecn]

for ai in [1,2]:
    anim = np.swapaxes(animations[ai][0].copy(), 0, 1)
    
    joints, root_x, root_z, root_r = anim[:,:-7], anim[:,-7], anim[:,-6], anim[:,-5]
    joints = joints.reshape((len(joints), -1, 3))
    
    rotation = Quaternions.id(1)
    offsets = []
    translation = np.array([[0,0,0]])
    
    #if not ignore_root:
    for i in range(len(joints)):
        joints[i,:,:] = rotation * joints[i]
        joints[i,:,0] = joints[i,:,0] + translation[0,0]
        joints[i,:,2] = joints[i,:,2] + translation[0,2]
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
        offsets.append(rotation * np.array([0,0,1]))
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])

    animations[ai] = joints

interval = 20

scale = 1.25*((len(animations))/2)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-scale*30, scale*30)
ax.set_zlim3d( 0, scale*60)
ax.set_ylim3d(-scale*30, scale*30)
#ax.set_xticks([], [])
#ax.set_yticks([], [])
#ax.set_zticks([], [])
ax.set_aspect('equal')

acolors = list(sorted(colors.cnames.keys()))[::-1]
lines = []

#  I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
#  J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
#  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
#parents = np.array([-1,0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20])
parents = [
    np.array([-1,  0,  1,  2, -1, -1,  0,  6,  7, -1, -1, -1,  0, 12, -1, 13, -1, 13, 17, 18, -1, -1, -1, -1, -1, 13, 25, 26, -1, -1, -1, -1]),
    np.array([-1,0,1,2,3, -1,1,6,7,-1,1,10,11,12,12,14,15,-1,12,18,19,-1]),
    np.array([-1,0,1,2,3, 4,1,6,7, 8,1,10,11,12,12,14,15,16,12,18,19,20]),
]

for ai, anim in enumerate(animations):
    lines.append([plt.plot([0,0], [0,0], [0,0],
                           color=acolors[ai],
                           lw=2,
                           path_effects=[pe.Stroke(linewidth=3, foreground='black'),
                                         pe.Normal()],
                           )[0] for _ in range(anim.shape[1])])

def animate(i):
    changed = []
    
    for ai in range(len(animations)):
    
        offset = 25*(ai-((len(animations))/2))
    
        for j in range(len(parents[ai])):
            
            if parents[ai][j] != -1:
                lines[ai][j].set_data(
                    [ animations[ai][i,j,0]+offset, animations[ai][i,parents[ai][j],0]+offset],
                    [-animations[ai][i,j,2],       -animations[ai][i,parents[ai][j],2]])
                lines[ai][j].set_3d_properties(
                    [ animations[ai][i,j,1],        animations[ai][i,parents[ai][j],1]])
        
        changed += lines[ai]
        
    return changed
    
plt.tight_layout()
    
ani = animation.FuncAnimation(fig, 
    animate, np.arange(len(animations[0])), interval=interval)


plt.show()
    
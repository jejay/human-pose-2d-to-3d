import numpy as np
import tensorflow as tf
import gc
import os
import sys
sys.path.append("../libs")
from Quaternions import Quaternions

DATA_PATH = "../data/manifold/processed/"

Xcmu = np.load(DATA_PATH + 'data_cmu.npz')['clips']
Xcmu = Xcmu.astype(np.float32, copy=False)
np.savez('data_cmu_32.npz', clips=Xcmu)
del Xcmu
gc.collect()

Xhdm05 = np.load(DATA_PATH + 'data_hdm05.npz')['clips']
Xhdm05 = Xhdm05.astype(np.float32, copy=False)
np.savez('data_hdm05_32.npz', clips=Xhdm05)
del Xhdm05
gc.collect()

Xmhad = np.load(DATA_PATH + 'data_mhad.npz')['clips']
Xmhad = Xmhad.astype(np.float32, copy=False)
np.savez('data_mhad_32.npz', clips=Xmhad)
del Xmhad
gc.collect()

Xedin_locomotion = np.load(DATA_PATH + 'data_edin_locomotion.npz')['clips']
Xedin_locomotion = Xedin_locomotion.astype(np.float32, copy=False)
np.savez('data_edin_locomotion_32.npz', clips=Xedin_locomotion)
del Xedin_locomotion
gc.collect()

Xedin_xsens = np.load(DATA_PATH + 'data_edin_xsens.npz')['clips']
Xedin_xsens = Xedin_xsens.astype(np.float32, copy=False)
np.savez('data_edin_xsens_32.npz', clips=Xedin_xsens)
del Xedin_xsens
gc.collect()

Xedin_misc = np.load(DATA_PATH + 'data_edin_misc.npz')['clips']
Xedin_misc = Xedin_misc.astype(np.float32, copy=False)
np.savez('data_edin_misc_32.npz', clips=Xedin_misc)
del Xedin_misc
gc.collect()

Xedin_punching = np.load(DATA_PATH + 'data_edin_punching.npz')['clips']
Xedin_punching = Xedin_punching.astype(np.float32, copy=False)
np.savez('data_edin_punching_32.npz', clips=Xedin_punching)
del Xedin_punching
gc.collect()

X=np.concatenate([
    np.load('data_cmu_32.npz')['clips'],
    np.load('data_hdm05_32.npz')['clips'],
    np.load('data_mhad_32.npz')['clips'],
    np.load('data_edin_locomotion_32.npz')['clips'],
    np.load('data_edin_xsens_32.npz')['clips'],
    np.load('data_edin_misc_32.npz')['clips'],
    np.load('data_edin_punching_32.npz')['clips']
], axis=0)

os.remove('data_cmu_32.npz')
os.remove('data_hdm05_32.npz')
os.remove('data_mhad_32.npz')
os.remove('data_edin_locomotion_32.npz')
os.remove('data_edin_xsens_32.npz')
os.remove('data_edin_misc_32.npz')
os.remove('data_edin_punching_32.npz')

#X = np.swapaxes(X, 1, 2)

np.random.shuffle(X)

writer = tf.python_io.TFRecordWriter('../data/tfrecords/cmu.tfrecords')

RELEVANT_2D_JOINTS = np.array([4,3,2,6,7,8,13,16,15,14,18,19,20])
# RFoot RKnee RHip LHip LKnee LFoot Head RWrist RElbow RShoulder LShoulder LElbow LWrist

for x in X:
    joints, root_x, root_z, root_r = x[:,:-7].copy(), x[:,-7].copy(), x[:,-6].copy(), x[:,-5].copy()
    joints = joints.reshape((len(joints), -1, 3))
    joints = joints[:,RELEVANT_2D_JOINTS,:]
    
    rotation = Quaternions.id(1)
    
    for j in range(len(joints)):
        joints[j,:,:] = rotation * joints[j]
        rotation = Quaternions.from_angle_axis(-root_r[j], np.array([0,1,0])) * rotation

    joints[:, :, 1] -= np.max(joints[:, :, 1:2], axis=1)/2
    
    writer.write(tf.train.Example(
        features=tf.train.Features(feature={
            's': tf.train.Feature(float_list=tf.train.FloatList(value=np.concatenate([x, joints.reshape((len(joints), -1))], axis=1).flatten()))
        })
    ).SerializeToString())
import numpy as np
import tensorflow as tf
from SingleManifoldModel import SingleManifoldModel
from AnimationPlot import animation_plot

tf.reset_default_graph()
rng = np.random.RandomState(23455)

params = np.load("network_core.npz")
preprocess = np.load('preprocess_core.npz')

X = np.load('data/processed/data_cmu.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(np.float32)
X = (X - preprocess['Xmean']) / preprocess['Xstd']

inputs = tf.placeholder(dtype=tf.float32, shape=[None, 73, X.shape[2]]) # (batchsize, 73, window)
model = SingleManifoldModel(window=X.shape[2])
model.build_graph(inputs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

index = 1328 #rng.randint(X.shape[0])
Xorgi = np.array(X[index:index+1])
Xnois = ((Xorgi * rng.binomial(size=Xorgi.shape, n=1, p=0.5)) / 0.5).astype(np.float32)
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

#Xrecn[:,-7:-4] = Xorgi[:,-7:-4]
    
ani = animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)

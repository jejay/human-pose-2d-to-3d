import numpy as np
import tensorflow as tf
from SingleManifoldModel import SingleManifoldModel
from AnimationPlot import animation_plot
import gc
import os.path
import os

rng = np.random.RandomState(23456)
tf.reset_default_graph()

if not os.path.isfile('data.tfrecords'):
    Xcmu = np.load('data/processed/data_cmu.npz')['clips']
    Xcmu = Xcmu.astype(np.float32, copy=False)
    np.savez('data_cmu_32.npz', clips=Xcmu)
    del Xcmu
    gc.collect()
    
    Xhdm05 = np.load('data/processed/data_hdm05.npz')['clips']
    Xhdm05 = Xhdm05.astype(np.float32, copy=False)
    np.savez('data_hdm05_32.npz', clips=Xhdm05)
    del Xhdm05
    gc.collect()
    
    Xmhad = np.load('data/processed/data_mhad.npz')['clips']
    Xmhad = Xmhad.astype(np.float32, copy=False)
    np.savez('data_mhad_32.npz', clips=Xmhad)
    del Xmhad
    gc.collect()
    
    Xedin_locomotion = np.load('data/processed/data_edin_locomotion.npz')['clips']
    Xedin_locomotion = Xedin_locomotion.astype(np.float32, copy=False)
    np.savez('data_edin_locomotion_32.npz', clips=Xedin_locomotion)
    del Xedin_locomotion
    gc.collect()
    
    Xedin_xsens = np.load('data/processed/data_edin_xsens.npz')['clips']
    Xedin_xsens = Xedin_xsens.astype(np.float32, copy=False)
    np.savez('data_edin_xsens_32.npz', clips=Xedin_xsens)
    del Xedin_xsens
    gc.collect()
    
    Xedin_misc = np.load('data/processed/data_edin_misc.npz')['clips']
    Xedin_misc = Xedin_misc.astype(np.float32, copy=False)
    np.savez('data_edin_misc_32.npz', clips=Xedin_misc)
    del Xedin_misc
    gc.collect()
    
    Xedin_punching = np.load('data/processed/data_edin_punching.npz')['clips']
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

    X = np.swapaxes(X, 1, 2)
    
    preprocess = np.load('preprocess_core.npz')
    X = (X - preprocess['Xmean']) / preprocess['Xstd']
    np.random.shuffle(X)

    writer = tf.python_io.TFRecordWriter('data.tfrecords')
    for x in X:
        writer.write(tf.train.Example(
            features=tf.train.Features(feature={
                's': tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten()))
            })
        ).SerializeToString())

dataset = tf.data.TFRecordDataset('data.tfrecords')
#dataset = dataset.shuffle(buffer_size=256)

def parser(record):
    parsed = tf.parse_single_example(record, {
        "s": tf.FixedLenFeature((17520), tf.float32),
    })
    return tf.reshape(parsed["s"], [73, 240])

dataset = dataset.map(parser)
dataset = dataset.batch(64)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = SingleManifoldModel(window=240)
model.build_graph(next_element)

loss = tf.losses.mean_squared_error(labels=next_element, predictions=model.autoencoded)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(1):
  sess.run(iterator.initializer)
  print("start epoch #", epoch+1)
  while True:
    try:
      l,t = sess.run((loss,train))
      print("loss: ", l)
    except tf.errors.OutOfRangeError:
      break

print("done")

preprocess = np.load('preprocess_core.npz')

X = np.load('data/processed/data_cmu.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(np.float32)
X = (X - preprocess['Xmean']) / preprocess['Xstd']

index = 1328 #rng.randint(X.shape[0])
Xorgi = np.array(X[index:index+1])
Xnois = ((Xorgi * rng.binomial(size=Xorgi.shape, n=1, p=0.5)) / 0.5).astype(np.float32)
Xrecn = np.array(sess.run(model.autoencoded, {
    next_element: Xnois
}))    

Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    
ani = animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
import numpy as np
import tensorflow as tf
from DeepManifoldModel import DeepManifoldModel

tf.reset_default_graph()

data = np.load("h36m.npz")

X = np.array(data["X"])

with tf.device("/cpu:0"):
    frames = tf.placeholder(tf.float32)

batchsize = 32

def noise_data(tensor_in):
    tensor_in_noised = tensor_in + tf.random_normal(shape=tf.shape(tensor_in), mean=0.0, stddev=0.1, dtype=tf.float32)
    joints = tf.reshape(tf.slice(tensor_in_noised, [0, 0, 0], [-1, -1, 3*24]), shape=[-1, 256, 24, 3]) ##TRANSPOSE FIRST
    masked_joints = tf.layers.dropout(joints, rate=0.25, noise_shape=[tf.shape(tensor_in)[0], 1, 24, 1], training=True)
    
    return (tf.transpose(tf.concat([
        tf.reshape(masked_joints, shape=[-1, 256, 3*24]),
        tf.reshape(tf.slice(tensor_in, [0, 0, 3*24], [-1, -1, 1]), shape=[-1, 256, 1]), # one/zero line
        tf.zeros(shape=[tf.shape(tensor_in)[0], 256, 4])
    ], axis=2), [0, 2, 1]), tf.transpose(tensor_in, [0, 2, 1]))

def unit(tensor_in):
    return (tf.transpose(tensor_in, [0, 2, 1]), tf.transpose(tensor_in, [0, 2, 1]))

dataset = tf.data.Dataset.from_tensor_slices(frames)
dataset = dataset.batch(batchsize)
dataset = dataset.map(noise_data)
datset = dataset.prefetch(1)
iterator = dataset.make_initializable_iterator()
next_input, next_label = iterator.get_next()

model = DeepManifoldModel(window=256)
model.build_graph(next_input)

loss = tf.losses.mean_squared_error(labels=next_label, predictions=model.autoencoded)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "checkpoints/h36m.ckpt")
#sess.run(tf.global_variables_initializer())

losses = []
for epoch in range(200):
    sess.run(iterator.initializer, {
        frames: X,
        model.training: True
    })
    losses.append([])
    print("start epoch #", epoch+1)
    while True:
        try:
            l,t = sess.run((loss,train))
            losses[len(losses)-1].append(l)
        except tf.errors.OutOfRangeError:
            break
    print("mean loss: ", np.mean(losses[len(losses)-1]))
print("done")

#preprocess = np.load('preprocess_core.npz')

#X = np.load(DATA_PATH + 'data_cmu.npz')['clips']
#X = np.swapaxes(X, 1, 2).astype(np.float32)
#X = (X - preprocess['Xmean']) / preprocess['Xstd']

#index = 1328 #rng.randint(X.shape[0])
#Xorgi = np.array(X[index:index+1])
#Xnois = ((Xorgi * rng.binomial(size=Xorgi.shape, n=1, p=0.5)) / 0.5).astype(np.float32)
#Xrecn = np.array(sess.run(model.autoencoded, {
#    next_element: Xnois
#}))"""
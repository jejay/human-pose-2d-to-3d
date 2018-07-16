import numpy as np
import tensorflow as tf
from DeepManifoldModel import DeepManifoldModel

tf.reset_default_graph()

BATCHSIZE = 32
WINDOWSIZE = 256
NUMJOINTS = 24

def noise_data(tensor_in):
    tensor_in_noised = tensor_in + tf.random_normal(shape=tf.shape(tensor_in), mean=0.0, stddev=0.1, dtype=tf.float32)
    
    joints = tf.reshape(tf.slice(tensor_in_noised,
                                 [0, 0, 0],
                                 [-1, -1, 3*NUMJOINTS]),
                        shape=[-1, WINDOWSIZE, 24, 3]) ##TRANSPOSE FIRST
    
    masked_joints = tf.layers.dropout(joints,
                                      rate=0.25,
                                      noise_shape=[tf.shape(tensor_in)[0], 1, NUMJOINTS, 1],
                                      training=True)
    
    return (tf.transpose(tf.concat([
        tf.reshape(masked_joints, shape=[-1, WINDOWSIZE, 3*NUMJOINTS]),
        tf.reshape(tf.slice(tensor_in, [0, 0, 3*NUMJOINTS], [-1, -1, 1]), shape=[-1, WINDOWSIZE, 1]), # one/zero line
        tf.zeros(shape=[tf.shape(tensor_in)[0], WINDOWSIZE, 4])
    ], axis=2), [0, 2, 1]), tf.transpose(tensor_in, [0, 2, 1]))

tfrecordsfile = tf.placeholder(tf.string)

dataset = tf.data.TFRecordDataset(tfrecordsfile)
dataset = dataset.shuffle(buffer_size=32)
dataset = dataset.map(lambda record: tf.reshape(
        tf.parse_single_example(record,
            {
                "feature": tf.FixedLenFeature((WINDOWSIZE*77), tf.float32)
            }
        )["feature"],
        [WINDOWSIZE, 77]
    )
)
dataset = dataset.batch(BATCHSIZE)
dataset = dataset.map(noise_data)
datset = dataset.prefetch(1)
iterator = dataset.make_initializable_iterator()
next_input, next_label = iterator.get_next()

model = DeepManifoldModel(window=256)
model.build_graph(next_input)

loss = tf.losses.mean_squared_error(labels=next_label, predictions=model.autoencoded)
tf.summary.scalar("loss", loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

saver = tf.train.Saver(max_to_keep=5)
sess = tf.Session()
#saver.restore(sess, "checkpoints/model.ckpt")
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
valid_writer = tf.summary.FileWriter('logs/valid')
sess.run(tf.global_variables_initializer())

losses = []
train_step = 0
valid_step = 0
for epoch in range(250):
    print("start epoch #", epoch+1)
    sess.run(iterator.initializer, {
        tfrecordsfile: '../../data/tfrecords/h36m-train.tfrecords',
    })
    while True:
        try:
            summary, _ = sess.run((merged, train), {model.training: True})
            train_step += 1
            train_writer.add_summary(summary, train_step)
        except tf.errors.OutOfRangeError:
            break

    sess.run(iterator.initializer, {
        tfrecordsfile: '../../data/tfrecords/h36m-valid.tfrecords',
    })
    while True:
        try:
            summary = sess.run(merged, {model.training: False})
            valid_step += 1
            valid_writer.add_summary(summary, valid_step)
        except tf.errors.OutOfRangeError:
            break
    
    if epoch % 10 == 9:
        save_path = saver.save(sess, "checkpoints/model.ckpt", global_step=epoch)
print("done")

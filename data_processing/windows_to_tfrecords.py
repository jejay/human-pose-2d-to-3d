import numpy as np
import tensorflow as tf

X = np.load("../data/windows/h36m.npz")["Xtrain"]

writer = tf.python_io.TFRecordWriter('../data/tfrecords/h36m-train.tfrecords')

for i in list(range(len(X))):
    writer.write(tf.train.Example(
        features=tf.train.Features(feature={
            'feature': tf.train.Feature(float_list=tf.train.FloatList(value=X[i].flatten()))
        })
    ).SerializeToString())
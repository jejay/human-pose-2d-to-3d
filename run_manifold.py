import sys
sys.path.append("model")
sys.path.append("libs")
import os
import numpy as np
import tensorflow as tf
from FlexibleSerialManifoldModel import FlexibleSerialManifoldModel
from OrnsteinUhlenbeckNoise import ornsteinUhlenbeckNoise
from SingleManifoldModel import SingleManifoldModel
from ManifoldModel import ManifoldModel


tf.app.flags.DEFINE_float("start_learning_rate", 1e-4, "Learning rate at the first epoch")
tf.app.flags.DEFINE_float("end_learning_rate", 1e-5, "Learning rate at the last epoch")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 30, "How many epochs we should train for")

# Architecture
tf.app.flags.DEFINE_integer("window_length", 240, "Number of frames in a window (motion clip).")
tf.app.flags.DEFINE_string("architecture", "linear-hourglass", "Architecture to use.")
tf.app.flags.DEFINE_boolean("procrustes", False, "Whether to rotational align 2D projections.")
tf.app.flags.DEFINE_boolean("independent", False, "Whether to draw the camera angles independent of time.")
tf.app.flags.DEFINE_boolean("toy", False, "Whether to use the same toy matrice/camera angle for all poses.")
tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")


# Data
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "experiments", "Training directory.")
tf.app.flags.DEFINE_string("dataset", "cmu", "Data directory")

# Train or load
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("save_predictions", False, "Set to True for saving predictions.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.join(FLAGS.train_dir,
                         "__".join(['data_{0}'.format(FLAGS.dataset),
                                    "architecture_{0}".format(FLAGS.architecture),
                                    'toy_on' if FLAGS.toy else 'toy_off',
                                    'independent_on' if FLAGS.independent else 'independent_off',
                                    'procrustes_on' if FLAGS.procrustes else 'procrustes_off',
                                    'bnorm_on' if FLAGS.batch_norm else 'bnorm_off']))

os.system('mkdir -p {}'.format(os.path.join( train_dir, "logs" ))) # To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448
os.system('mkdir -p {}'.format(os.path.join( train_dir, "checkpoints" ))) # To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448

def main(_):
    tf.reset_default_graph()
    
        
    preprocess_2d = np.load("data/preprocess_2d.npz")
    preprocess_core = np.load("data/preprocess_core.npz")

    core_mean = tf.constant(preprocess_core["Xmean"].reshape((73)), dtype=tf.float32)
    core_std = tf.constant(preprocess_core["Xstd"].reshape((73)), dtype=tf.float32)
    
    np_mean = tf.constant(preprocess_2d["np_mean"], dtype=tf.float32)
    np_std = tf.constant(preprocess_2d["np_std"], dtype=tf.float32)

    p_mean = tf.constant(preprocess_2d["p_mean"], dtype=tf.float32)
    p_std = tf.constant(preprocess_2d["p_std"], dtype=tf.float32)

    learning_rate = tf.placeholder(dtype=tf.float32)
    independent = tf.placeholder(tf.bool)
    procrustes = tf.placeholder(tf.bool)
    tfrecordsfile = tf.placeholder(tf.string)
    cams = tf.constant(np.load(os.path.join(FLAGS.data_dir, "cammats.npz"))["mats"])
    toycam = tf.constant([[ 6.123234e-17,  0.000000e+00, -1.000000e+00],
                          [ 0.000000e+00,  1.000000e+00,  0.000000e+00],
                          [ 1.000000e+00,  0.000000e+00,  6.123234e-17]], dtype=tf.float32)
    
    def project_to_2d(x):
        x_2d = tf.slice(x, [0,0,73], [-1, -1, -1])
        
        if FLAGS.toy:
            x_2d = tf.reshape(x_2d, [-1, 3])
            x_2d @= toycam
        else:
            x_2d_indep = tf.reshape(x_2d, [-1, 13, 3])
            x_2d_indep @= tf.random_shuffle(cams)[0:tf.shape(x)[0]*FLAGS.window_length]
            
            x_2d_cor = tf.reshape(x_2d, [-1, FLAGS.window_length * 13, 3])
            x_2d_cor @= tf.random_shuffle(cams)[0:tf.shape(x)[0]]
            
            x_2d = tf.cond(independent, lambda: x_2d_indep, lambda: x_2d_cor)
        
        x_2d = tf.reshape(x_2d, [-1, FLAGS.window_length, 13, 3])
        depth = tf.slice(x_2d, [0, 0, 0, 2], [-1, -1, -1, -1])
        x_2d = tf.slice(x_2d, [0, 0, 0, 0], [-1, -1, -1, 2])
        f = tf.random_uniform(shape=[], minval=16, maxval=128)
        x_2d /= (f+depth)
        
        x_3d = tf.slice(x, [0,0,0], [-1, -1, 73])
        x_3d = (x_3d - core_mean) / core_std
        
        return (x_2d, x_3d)
    
    def normalize(x_2d, x_3d):
        hips = (tf.slice(x_2d, [0,0,2,0], [-1, -1, 1, -1]) + tf.slice(x_2d, [0,0,3,0], [-1, -1, 1, -1])) / 2
        necks = (tf.slice(x_2d, [0,0,9,0], [-1, -1, 1, -1]) + tf.slice(x_2d, [0,0,10,0], [-1, -1, 1, -1])) / 2
        
        x_2d = x_2d - hips
        x_2d /= tf.norm(hips-necks, axis=3)[:, :, None, :]
        
        hipsNormalized = (tf.slice(x_2d, [0,0,2,0], [-1, -1, 1, -1]) + tf.slice(x_2d, [0,0,3,0], [-1, -1, 1, -1])) / 2
        necksNormalized = (tf.slice(x_2d, [0,0,9,0], [-1, -1, 1, -1]) + tf.slice(x_2d, [0,0,10,0], [-1, -1, 1, -1])) / 2
        hipsnecksT = tf.reshape(tf.concat([hipsNormalized, necksNormalized], axis=2), [-1,2,2])
        B = tf.constant(np.array([[0,0],[0,1]]).T.astype(np.float32))
        M = tf.reshape(tf.transpose(tf.reshape(B @ tf.reshape(tf.transpose(hipsnecksT, [1,0,2]), [2, -1]), [2, -1, 2]), [1,0,2]), [-1, FLAGS.window_length, 2,2])
        s, U, V = tf.svd(M, full_matrices=True)
        R = tf.matmul(U, V, adjoint_b=True)
        x_2d = tf.cond(procrustes, lambda: x_2d @ tf.transpose(R, [0,1,3,2]), lambda: x_2d)
        x_2d = tf.reshape(x_2d, [-1, FLAGS.window_length, 26])
        x_2d = tf.cond(procrustes, lambda: (x_2d - p_mean) / p_std, lambda: (x_2d - np_mean) / np_std)
        return (tf.transpose(x_2d, [0,2,1]), tf.transpose(x_3d, [0,2,1]))
    
    dataset = tf.data.TFRecordDataset(tfrecordsfile)
    dataset = dataset.map(lambda record: tf.reshape(
            tf.parse_single_example(record,
                {
                    "s": tf.FixedLenFeature((FLAGS.window_length*(112)), tf.float32)
                }
            )["s"],
            [FLAGS.window_length, 112]
        )
    )
    
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.map(project_to_2d)
    dataset = dataset.map(normalize)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_initializable_iterator()
    x_2d_input, x_3d = iterator.get_next()
    

    hourglass = ManifoldModel(window=FLAGS.window_length,
                      activation=None,
                      maxpooling=False,
                      accurate_depooling=False,
                      batch_norm=False,
                      residual=False)
    
    manifold_weights = np.load("data/network_core.npz")
    manifold = ManifoldModel(window=FLAGS.window_length,
                          maxpooling=True,
                          accurate_depooling=False,
                          batch_norm=False,
                          residual=False)
    
    if FLAGS.architecture in ['linear-hourglass', 'linear-half-hourglass']:
    
        x_2d = tf.transpose(x_2d_input, [0,2,1])
        x_2d = tf.reshape(x_2d, [-1,26])
        
        x_2d = tf.layers.dense(x_2d, 1024)
        skip = x_2d
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)
        
        x_2d = tf.layers.dropout(x_2d, rate=0.5, training=hourglass.training)
        x_2d = tf.layers.dense(x_2d, 1024)
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)
    
        x_2d = tf.layers.dropout(x_2d, rate=0.5, training=hourglass.training)
        x_2d = tf.layers.dense(x_2d, 1024)
        x_2d += skip
        
        x_2d = tf.reshape(x_2d, [-1, FLAGS.window_length, 1024])
        x_2d = tf.transpose(x_2d, [0,2,1])
        skip_from_linear = x_2d
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)

        if FLAGS.architecture == 'linear-hourglass':        
            x_2d = hourglass.conv_in(x_2d, channels_in=1024, channels_out=512, kernel_length=21, dropout=0.2)
            skip1 = x_2d
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_in(x_2d, channels_in=512, channels_out=512, kernel_length=21, dropout=0.2)
            skip2 = x_2d
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_in(x_2d, channels_in=512, channels_out=512, kernel_length=21, dropout=0.2)
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_out(x_2d, channels_in=512, channels_out=512, kernel_length=21, dropout=0.2)
            x_2d += skip2
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_out(x_2d, channels_in=512, channels_out=512, kernel_length=21, dropout=0.2)
            x_2d += skip1
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_out(x_2d, channels_in=512, channels_out=256, kernel_length=21, dropout=0.2)
            x_2d = tf.nn.relu(x_2d)

        elif FLAGS.architecture == 'linear-half-hourglass':
            x_2d = tf.layers.dropout(x_2d, rate=0.2, training=hourglass.training)
            x_2d = tf.layers.conv1d(
                tf.reshape(x_2d, [-1, 1024, FLAGS.window_length]),
                filters=1024,
                kernel_size=45,
                strides=2,
                padding='same',
                data_format='channels_first'
            ) # (batchsize, channels_out, window)
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = tf.layers.dropout(x_2d, rate=0.2, training=hourglass.training)
            x_2d = tf.layers.conv1d(
                x_2d,
                filters=1024,
                kernel_size=45,
                strides=2,
                padding='same',
                data_format='channels_first'
            ) # (batchsize, channels_out, window)
            x_2d += tf.reshape(tf.image.resize_images(
                tf.reshape(skip_from_linear, [-1, 1024, FLAGS.window_length, 1]),
                [1024, FLAGS.window_length//4],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            ), [-1, 1024, FLAGS.window_length//4])
            x_2d = tf.nn.relu(x_2d)
    
    elif FLAGS.architecture == 'hourglass':
    
        with tf.variable_scope("hourglass"):
            x_2d = hourglass.conv_in(x_2d_input, channels_in=26, channels_out=512, kernel_length=21, dropout=0.2)
            skip1 = x_2d
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_in(x_2d, channels_in=512, channels_out=512, kernel_length=21, dropout=0.2)
            skip2 = x_2d
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_in(x_2d, channels_in=512, channels_out=512, kernel_length=21, dropout=0.2)
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_out(x_2d, channels_in=512, channels_out=512, kernel_length=21, dropout=0.2)
            x_2d += skip2
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_out(x_2d, channels_in=512, channels_out=512, kernel_length=21, dropout=0.2)
            x_2d += skip1
            if FLAGS.batch_norm:
                x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
            x_2d = tf.nn.relu(x_2d)
            
            x_2d = hourglass.conv_out(x_2d, channels_in=512, channels_out=256, kernel_length=21, dropout=0.2)
            x_2d = tf.nn.relu(x_2d)
    
    elif FLAGS.architecture in ['tower', 'eiffel-tower']:
        
        x_2d = tf.layers.dropout(x_2d_input, rate=0.2, training=hourglass.training)
        x_2d = tf.layers.conv1d(
            x_2d,
            filters=256 if FLAGS.architecture == 'tower' else 64,
            kernel_size=85 if FLAGS.architecture == 'tower' else 45,
            strides=1,
            padding='same',
            data_format='channels_first'
        ) # (batchsize, channels_out, window)
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)
        
        x_2d = tf.layers.dropout(x_2d, rate=0.2, training=hourglass.training)
        x_2d = tf.layers.conv1d(
            x_2d,
            filters=256 if FLAGS.architecture == 'tower' else 128,
            kernel_size=45 if FLAGS.architecture == 'tower' else 25,
            strides=1,
            padding='same',
            data_format='channels_first'
        ) # (batchsize, channels_out, window)
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)    
        
        x_2d = tf.layers.dropout(x_2d, rate=0.2, training=hourglass.training)
        x_2d = tf.layers.conv1d(
            x_2d,
            filters=256,
            kernel_size=45 if FLAGS.architecture == 'tower' else 15,
            strides=1,
            padding='same',
            data_format='channels_first'
        ) # (batchsize, channels_out, window)
        x_2d = tf.nn.relu(x_2d)
    
    
    elif FLAGS.architecture == 'monster-tower':

        x_2d = tf.layers.dropout(x_2d_input, rate=0.2, training=hourglass.training)
        x_2d = tf.layers.conv1d(
            x_2d,
            filters=1024,
            kernel_size=85,
            strides=1,
            padding='same',
            data_format='channels_first'
        ) # (batchsize, channels_out, window)
        skip = x_2d
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)
        
        x_2d = tf.layers.dropout(x_2d, rate=0.2, training=hourglass.training)
        x_2d = tf.layers.conv1d(
            x_2d,
            filters=1024,
            kernel_size=65,
            strides=1,
            padding='same',
            data_format='channels_first'
        ) # (batchsize, channels_out, window)
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)    
        
        x_2d = tf.layers.dropout(x_2d, rate=0.2, training=hourglass.training)
        x_2d = tf.layers.conv1d(
            x_2d,
            filters=1024,
            kernel_size=45,
            strides=1,
            padding='same',
            data_format='channels_first'
        ) # (batchsize, channels_out, window)
        x_2d += skip
        skip = x_2d
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)

        x_2d = tf.layers.dropout(x_2d, rate=0.2, training=hourglass.training)
        x_2d = tf.layers.conv1d(
            x_2d,
            filters=1024,
            kernel_size=25,
            strides=1,
            padding='same',
            data_format='channels_first'
        ) # (batchsize, channels_out, window)
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)

        x_2d = tf.layers.dropout(x_2d, rate=0.2, training=hourglass.training)
        x_2d = tf.layers.conv1d(
            x_2d,
            filters=1024,
            kernel_size=15,
            strides=1,
            padding='same',
            data_format='channels_first'
        ) # (batchsize, channels_out, window)
        x_2d += skip
        if FLAGS.batch_norm:
            x_2d = tf.layers.batch_normalization(x_2d, axis=1, momentum=0.99995, training=hourglass.training, fused=True)
        x_2d = tf.nn.relu(x_2d)

        x_2d = tf.layers.dropout(x_2d, rate=0.2, training=hourglass.training)
        x_2d = tf.layers.conv1d(
            x_2d,
            filters=256,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format='channels_first'
        ) # (batchsize, channels_out, window)
        x_2d = tf.nn.relu(x_2d)
    
    with tf.variable_scope("last-pooling"):
        x_2d, _ = tf.nn.max_pool_with_argmax(
            tf.reshape(x_2d, [-1, 256, FLAGS.window_length, 1]),
            ksize=[1,1,2,1],
            strides=[1,1,2,1],
            padding="VALID"
        ) # (batchsize, 256, 120, 1)
        x_2d = tf.reshape(x_2d, [-1, 256, FLAGS.window_length//2])
    


    with tf.variable_scope("half-manifold"):
        #manifold_hidden_units = manifold.conv_in(x_3d, channels_in=128, channels_out=256, kernel_length=25, dropout=0.2,
        #                         kernel_weights=manifold_weights['L000_L001_W'].transpose(),
        #                         bias_weights=manifold_weights['L000_L002_b'].reshape((256,)))
        manifold._depth = 2
        autoencoded = manifold.conv_out(x_2d, channels_in=256, channels_out=73, kernel_length=25, dropout=0.0, output_layer=True,
                                     kernel_weights=manifold_weights['L001_L002_W'].transpose(),
                                     bias_weights=manifold_weights['L001_L003_b'].reshape((73,)))

    
    
    
    
    
    loss = tf.losses.mean_squared_error(labels=x_3d, predictions=autoencoded)
    tf.summary.scalar("loss", loss)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = optimizer.minimize(loss)
    
    saver = tf.train.Saver(max_to_keep=2)

    print("TRAINABLE_VARIABLES:", tf.get_default_graph().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(train_dir, 'logs/train'), sess.graph)
    sess.run(tf.global_variables_initializer())
    print("number of parameters", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    train_step = 0
 
    maxepsqrt = FLAGS.epochs**0.5
    for epoch in range(FLAGS.epochs):
        epsqrt = epoch**0.5
        lr = FLAGS.start_learning_rate*((maxepsqrt-epsqrt)/maxepsqrt) + FLAGS.end_learning_rate*(epsqrt/maxepsqrt)
        print("start epoch #", epoch+1, "learning rate", lr)
        sess.run(iterator.initializer, {
            tfrecordsfile: os.path.join(FLAGS.data_dir, 'tfrecords', '{0}.tfrecords'.format(FLAGS.dataset)),
            independent: FLAGS.independent,
            procrustes: FLAGS.procrustes,
            hourglass.training: True,
            manifold.training: False
        })
        while True:
            try:
                _, summary = sess.run((train, merged), {
                    learning_rate: lr,
                    independent: FLAGS.independent,
                    procrustes: FLAGS.procrustes,
                    hourglass.training: True,
                    manifold.training: False
                })
                train_step += 1
                train_writer.add_summary(summary, train_step)
            except tf.errors.OutOfRangeError:
                break
        if epoch % 5 == 4:
            saver.save(sess, os.path.join(train_dir, "checkpoints/model.ckpt"), global_step=epoch)
    
    print("training done")
    print("start inferring results")
    sess.run(iterator.initializer, {
        tfrecordsfile: os.path.join(FLAGS.data_dir, 'tfrecords', '{0}.tfrecords'.format(FLAGS.dataset)),
        independent: FLAGS.independent,
        procrustes: FLAGS.procrustes,
        hourglass.training: False,
        manifold.training: False
    })
    l, result_input_x2d, result_output_x3d = sess.run((loss, x_2d_input, autoencoded), {
        learning_rate: lr,
        independent: FLAGS.independent,
        procrustes: FLAGS.procrustes,
        hourglass.training: False,
        manifold.training: False
    })
    
    result_input_x2d = result_input_x2d.swapaxes(1,2)
    if FLAGS.procrustes:
        result_input_x2d = result_input_x2d * preprocess_2d["p_std"] + preprocess_2d["p_mean"]
    else:
        result_input_x2d = result_input_x2d * preprocess_2d["np_std"] + preprocess_2d["np_mean"]
    result_output_x3d = result_output_x3d * preprocess_core["Xstd"] + preprocess_core["Xmean"]
    
    np.savez(os.path.join(train_dir, "results.npz"), x2d = result_input_x2d, x3d = result_output_x3d)

    print("inferring results done, loss was", l)

if __name__ == "__main__":
  tf.app.run()

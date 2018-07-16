import sys
sys.path.append("model")
sys.path.append("libs")
import os
import numpy as np
import tensorflow as tf
from FlexibleSerialManifoldModel import FlexibleSerialManifoldModel
from OrnsteinUhlenbeckNoise import ornsteinUhlenbeckNoise


tf.app.flags.DEFINE_float("start_learning_rate", 1e-4, "Learning rate at the first epoch")
tf.app.flags.DEFINE_float("end_learning_rate", 1e-5, "Learning rate at the last epoch")
tf.app.flags.DEFINE_float("dropout", 0.2, "Dropout rate. 0 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 300, "How many epochs we should train for")
tf.app.flags.DEFINE_boolean("batch_norm", False, "Use batch_normalization")

# Architecture
tf.app.flags.DEFINE_integer("window_length", 256, "Number of frames in a window (motion clip).")
tf.app.flags.DEFINE_integer("num_joints", 24, "Number of joints in a pose.")
tf.app.flags.DEFINE_integer("num_manifolds", 1, "Depth of the manifold, i.e. half of the number of total layers.")
tf.app.flags.DEFINE_integer("manifold_depth", 3, "Depth of the manifold, i.e. half of the number of total layers.")
tf.app.flags.DEFINE_integer("filter_length", 21, "Filter length of each 1D convolution.")
tf.app.flags.DEFINE_integer("conv_channels", 128, "Number of output channels of the first convolution. Number is doubled every further convolution.")
tf.app.flags.DEFINE_boolean("intermediate_supervision", False, "Whether to use intermediate supervision on each manifold output.")
tf.app.flags.DEFINE_boolean("intra_residual", True, "Whether to add intra residual connections.")
tf.app.flags.DEFINE_boolean("inter_residual", False, "Whether to add inter residual connections.")
tf.app.flags.DEFINE_boolean("maxpooling", True, "Whether to use maxpooling or just a convolution with stride 2 for downsampling.")
tf.app.flags.DEFINE_boolean("accurate", False, "Whether to place the number where it came from in depooling or just use nearest neighboar interpolation for upsamling.")
tf.app.flags.DEFINE_float("ou", 0.0, "Ornstein-Uhlenbeck training noise sigma.")
tf.app.flags.DEFINE_float("nn", 0.25, "Normal training noise sigma.")
tf.app.flags.DEFINE_float("mask", 0.33, "Joint mask out training probability.")
tf.app.flags.DEFINE_boolean("fldout", False, "Apply dropout to first layer.")

# Data
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "experiments", "Training directory.")
tf.app.flags.DEFINE_string("dataset", "h36m", "Data directory")

# Train or load
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("save_predictions", False, "Set to True for saving predictions.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.join(FLAGS.train_dir,
                         "__".join(['data_{0}'.format(FLAGS.dataset),
                                    'num_{0}'.format(FLAGS.num_manifolds),
                                    'depth_{0}'.format(FLAGS.manifold_depth),
                                    'window_{0}'.format(FLAGS.window_length),
                                    'filter_{0}'.format(FLAGS.filter_length),
                                    'slr_{0}'.format(FLAGS.start_learning_rate),
                                    'elr_{0}'.format(FLAGS.end_learning_rate),
                                    'channels_{0}'.format(FLAGS.conv_channels),
                                    'isv_on' if FLAGS.intermediate_supervision else 'isv_off',
                                    'intra_on' if FLAGS.intra_residual else 'intra_off',
                                    'inter_on' if FLAGS.inter_residual else 'inter_off',
                                    'batch_{0}'.format(FLAGS.batch_size),
                                    'bnorm_on' if FLAGS.batch_norm else 'bnorm_off',
                                    'mpool_on' if FLAGS.maxpooling else 'mpool_off',
                                    'acc_on' if FLAGS.accurate else 'acc_off',
                                    'ou_{0}'.format(FLAGS.ou),
                                    'nn_{0}'.format(FLAGS.nn),
                                    'mask_{0}'.format(FLAGS.mask),
                                    'fldout_on' if FLAGS.fldout else 'fldout_off']))

os.system('mkdir -p {}'.format(os.path.join( train_dir, "logs" ))) # To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448
os.system('mkdir -p {}'.format(os.path.join( train_dir, "checkpoints" ))) # To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448

def main(_):
    tf.reset_default_graph()
    
    preprocess = np.load("preprocess.npz")
    mean = tf.constant(preprocess['Xmean'], dtype=tf.float32)
    std = tf.constant(preprocess['Xstd'], dtype=tf.float32)

    normal_noise = tf.placeholder_with_default(0.0, [])
    ou_noise = tf.placeholder_with_default(0.0, [])
    mask_joint_prob = tf.placeholder_with_default(0.0, [])
    learning_rate = tf.placeholder(dtype=tf.float32)
    
    def preprocess_data(tensor_in):    
        x = tf.reshape(tf.slice(tensor_in,
                                [0, 0, 0],
                                [-1, -1, 3*FLAGS.num_joints]),
                                shape=[-1, FLAGS.window_length, FLAGS.num_joints, 3]) ##TRANSPOSE FIRST
    
        x = tf.layers.dropout(x,
                              rate=mask_joint_prob,
                              noise_shape=[tf.shape(tensor_in)[0], 1, FLAGS.num_joints, 1],
                              training=mask_joint_prob > 0.000001)

        x = tf.reshape(x,
                       shape=[-1, FLAGS.window_length, 3*FLAGS.num_joints])
        
        x = (x - tf.slice(mean, [0], [3*FLAGS.num_joints])) / tf.slice(std, [0], [3*FLAGS.num_joints])
                
        x = tf.cond(normal_noise > 0.000001,
                    true_fn=lambda: x + tf.random_normal(shape=tf.shape(x), stddev=normal_noise),
                    false_fn=lambda: x)

        x = tf.cond(ou_noise > 0.000001,
                    true_fn=lambda: x + ornsteinUhlenbeckNoise(n=3*FLAGS.num_joints, window=FLAGS.window_length, sigma=ou_noise),
                    false_fn=lambda: x)
        
        return (tf.transpose(tf.concat([
            x,
            tf.slice(tensor_in, [0, 0, 3*FLAGS.num_joints], [-1, -1, 1]), # one/zero line
            #tf.zeros(shape=[tf.shape(tensor_in)[0], FLAGS.window_length, 4]) #not necessary
        ], axis=2), [0, 2, 1]), tf.transpose((tensor_in - mean) / std, [0, 2, 1]))
    
    tfrecordsfile = tf.placeholder(tf.string)
    
    dataset = tf.data.TFRecordDataset(tfrecordsfile)
    dataset = dataset.map(lambda record: tf.reshape(
            tf.parse_single_example(record,
                {
                    "feature": tf.FixedLenFeature((FLAGS.window_length*(FLAGS.num_joints*3 + 5)), tf.float32)
                }
            )["feature"],
            [FLAGS.window_length, FLAGS.num_joints*3 + 5]
        )
    )
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.map(preprocess_data)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_initializable_iterator()
    next_input, next_label = iterator.get_next()
    
    model = FlexibleSerialManifoldModel(window=FLAGS.window_length,
                                        data_channels_in=FLAGS.num_joints*3 + 1,
                                        data_channels_out=FLAGS.num_joints*3 + 5,
                                        manifold_depth=FLAGS.manifold_depth,
                                        num_manifolds=FLAGS.num_manifolds,
                                        filter_length=FLAGS.filter_length,
                                        conv_channels=FLAGS.conv_channels,
                                        dropout=FLAGS.dropout,
                                        batch_norm=FLAGS.batch_norm,
                                        intra_residual=FLAGS.intra_residual,
                                        inter_residual=FLAGS.inter_residual,
                                        maxpooling=FLAGS.maxpooling,
                                        accurate_depooling=FLAGS.accurate,
                                        first_layer_dropout=FLAGS.fldout)
    model.build_graph(next_input)
    
    if FLAGS.intermediate_supervision:
        loss = tf.zeros_like(next_input)
        for i, supervision in enumerate(model.autoencoded):
            partial_loss = tf.losses.mean_squared_error(labels=next_label, predictions=supervision)
            loss += partial_loss
            tf.summary.scalar("partial_loss_{0}".format(i), loss)
    else:
        loss = tf.losses.mean_squared_error(labels=next_label, predictions=model.autoencoded[len(model.autoencoded)-1])
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = optimizer.minimize(loss)
    
    saver10 = tf.train.Saver(max_to_keep=3)
    saver50 = tf.train.Saver(max_to_keep=3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(train_dir, 'logs/train'), sess.graph)
    valid_nn_writer = tf.summary.FileWriter(os.path.join(train_dir, 'logs/valid-nn'))
    valid_ou_writer = tf.summary.FileWriter(os.path.join(train_dir, 'logs/valid-ou'))
    valid_vel_writer = tf.summary.FileWriter(os.path.join(train_dir, 'logs/valid-vel'))
    sess.run(tf.global_variables_initializer())
    print("number of parameters", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    train_step = 0
    valid_step = 0
    maxepsqrt = FLAGS.epochs**0.5
    for epoch in range(FLAGS.epochs):
        epsqrt = epoch**0.5
        lr = FLAGS.start_learning_rate*((maxepsqrt-epsqrt)/maxepsqrt) + FLAGS.end_learning_rate*(epsqrt/maxepsqrt)
        print("start epoch #", epoch+1, "learning rate", lr)
        sess.run(iterator.initializer, {
            tfrecordsfile: os.path.join(FLAGS.data_dir, 'tfrecords', '{0}-train.tfrecords'.format(FLAGS.dataset)),
            model.training: True,
            normal_noise: FLAGS.nn,
            ou_noise: FLAGS.ou,
            mask_joint_prob: FLAGS.mask,
            learning_rate: lr
        })
        while True:
            try:
                summary, _ = sess.run((merged, train), {
                    model.training: True,
                    normal_noise: FLAGS.nn,
                    ou_noise: FLAGS.ou,
                    mask_joint_prob: FLAGS.mask,
                    learning_rate: lr,
                })
                train_step += 1
                train_writer.add_summary(summary, train_step)
            except tf.errors.OutOfRangeError:
                break
    
        sess.run(iterator.initializer, {
            tfrecordsfile: os.path.join(FLAGS.data_dir, 'tfrecords', '{0}-valid.tfrecords'.format(FLAGS.dataset)),
            model.training: False,
            normal_noise: 0.25,
            ou_noise: 0,
            mask_joint_prob: 0.2
        })   
        local_valid_step = 0
        while True:
            try:
                summary = sess.run(merged, {
                    model.training: False,
                    normal_noise: 0.25,
                    ou_noise: 0,
                    mask_joint_prob: 0.2
                })
                local_valid_step += 1
                valid_nn_writer.add_summary(summary, valid_step + local_valid_step)
            except tf.errors.OutOfRangeError:
                break
    
        sess.run(iterator.initializer, {
            tfrecordsfile: os.path.join(FLAGS.data_dir, 'tfrecords', '{0}-valid.tfrecords'.format(FLAGS.dataset)),
            model.training: False,
            normal_noise: 0,
            ou_noise: 0.25,
            mask_joint_prob: 0.2
        })   
        local_valid_step = 0
        while True:
            try:
                summary = sess.run(merged, {
                    model.training: False,
                    normal_noise: 0,
                    ou_noise: 0.25,
                    mask_joint_prob: 0.2
                })
                local_valid_step += 1
                valid_ou_writer.add_summary(summary, valid_step + local_valid_step)
            except tf.errors.OutOfRangeError:
                break

        sess.run(iterator.initializer, {
            tfrecordsfile: os.path.join(FLAGS.data_dir, 'tfrecords', '{0}-valid.tfrecords'.format(FLAGS.dataset)),
            model.training: False,
            normal_noise: 0,
            ou_noise: 0,
            mask_joint_prob: 0
        })   
        local_valid_step = 0
        while True:
            try:
                summary = sess.run(merged, {
                    model.training: False,
                    normal_noise: 0,
                    ou_noise: 0,
                    mask_joint_prob: 0
                })
                local_valid_step += 1
                valid_vel_writer.add_summary(summary, valid_step + local_valid_step)
            except tf.errors.OutOfRangeError:
                break
        
        valid_step += local_valid_step
        
        if epoch % 50 == 49:
            saver50.save(sess, os.path.join(train_dir, "checkpoints/model.ckpt"), global_step=epoch)
        elif epoch % 10 == 9:
            saver10.save(sess, os.path.join(train_dir, "checkpoints/model.ckpt"), global_step=epoch)
    print("done")

if __name__ == "__main__":
  tf.app.run()

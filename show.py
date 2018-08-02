import sys
sys.path.append("model")
sys.path.append("libs")
import os
import numpy as np
import tensorflow as tf
from FlexibleSerialManifoldModel import FlexibleSerialManifoldModel
from OrnsteinUhlenbeckNoise import ornsteinUhlenbeckNoise
from Quaternions import Quaternions
from Pivots import Pivots
from AnimationPlotH36M import animation_plot


tf.app.flags.DEFINE_float("start_learning_rate", 1e-4, "Learning rate at the first epoch")
tf.app.flags.DEFINE_float("end_learning_rate", 1e-5, "Learning rate at the last epoch")
tf.app.flags.DEFINE_float("dropout", 0.2, "Dropout rate. 0 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
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
tf.app.flags.DEFINE_boolean("maxpooling", False, "Whether to use maxpooling or just a convolution with stride 2 for downsampling.")
tf.app.flags.DEFINE_boolean("accurate", False, "Whether to place the number where it came from in depooling or just use nearest neighboar interpolation for upsamling.")
tf.app.flags.DEFINE_float("ou", 0.0, "Ornstein-Uhlenbeck training noise sigma.")
tf.app.flags.DEFINE_float("nn", 0.1, "Normal training noise sigma.")
tf.app.flags.DEFINE_float("mask", 0.2, "Joint mask out training probability.")
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
plots = []
def main(_):
    
    poses3d = np.load("data/predictions/9.Photo.Photo 1.55011271.h5-sh.npz")["poses3d"]
    
    original_positions = np.array(poses3d).reshape(-1, 32, 3)#[0:10]
    positions = original_positions[:, np.array([
        #0, #Hips (Hip)
        1,  #RightUpLeg (RHip)
        2,  #RightLeg (RKnee)
        3,  #RightFoot (RFoot)
        0,#4,  #RightToeBase
        0,#5,  #Site
        6,  #LeftUpLeg (LHip)
        7,  #LeftLeg (LKnee)
        8,  #LeftFoot (LFoot)
        0,#9,  #LeftToeBase
        0,#10, #Site
        #11,#Spine
        12, #Spine1 (Spine)
        13, #Neck (Thorax)
        14, #Head (Neck/Nose)
        15, #Site (Head)
        #16,#LeftShoulder
        17, #LeftArm (LShoulder)
        18, #LeftForeArm (LElbow)
        #19,#LeftHand (LWrist)
        19,#20, #LeftHandThumb
        0,#21, #Site
        0,#22, #L_Wrist_End
        #23,#Site
        #24,#RightShoulder
        25, #RightArm (RShoulder)
        26, #RightForeArm (RElbow)
        #27,#RightHand (RWrist)
        27,#28, #RightHandThumb
        0,#29, #Site
        0,#30, #R_Wrist_End
        #31 #Site
    ])]#[0:10]
    
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
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1], forward="y", plane="xy").ps
    
    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, np.ones(shape=(len(positions), 1))], axis=-1)
    positions = np.concatenate([positions, np.zeros(shape=(len(positions), 3))], axis=-1)
    positions = np.concatenate([positions, rvelocity.reshape(-1, 1)], axis=-1)
    
    FLAGS.window_length = poses3d.shape[0]-1
    
    tf.reset_default_graph()
    
    input_sequence = tf.placeholder(dtype=tf.float32);
    
    preprocess = np.load("preprocess.npz")
    mean = tf.constant(preprocess['Xmean'], dtype=tf.float32)
    std = tf.constant(preprocess['Xstd'], dtype=tf.float32)

    normal_noise = tf.placeholder_with_default(0.0, [])
    ou_noise = tf.placeholder_with_default(0.0, [])
    mask_joint_prob = tf.placeholder_with_default(0.0, [])
    
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
    
    dataset = tf.data.Dataset.from_tensors(input_sequence)
    dataset = dataset.batch(1)
    dataset = dataset.map(preprocess_data)
    iterator = dataset.make_initializable_iterator()
    next_input, next_label = iterator.get_next()
    
    with tf.device("/gpu:0"):
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

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, os.path.join(train_dir, "checkpoints/model.ckpt-299"))
    print("number of parameters", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    
    sess.run(iterator.initializer, {
        input_sequence: positions,
        model.training: False,
        normal_noise: 0,
        ou_noise: 0,
        mask_joint_prob: 0
    })
    encoded = sess.run(model.autoencoded[len(model.autoencoded)-1], {
        model.training: False,
        normal_noise: 0,
        ou_noise: 0,
        mask_joint_prob: 0
    })
    
    encoded = np.array(encoded).swapaxes(1,2) * preprocess['Xstd'] + preprocess['Xmean']

    plots.append(animation_plot([positions, encoded[0]]))

if __name__ == "__main__":
  tf.app.run()

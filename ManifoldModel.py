import tensorflow as tf

class ManifoldModel(object):
    def __init__(self, window, activation=tf.nn.relu, accurate_depooling=False):
        self._window = window
        self._activation = activation
        self._pooling_ind_stack = []
        self._accurate_depooling = accurate_depooling
        
        self.training = tf.placeholder_with_default(False, [])
        
    def conv_in(self, x, channels_in, channels_out, kernel_length, dropout=0.25):
        droppedout = tf.layers.dropout(
            x,
            rate=dropout,
            seed=None,
            training=self.training
        )
        
        convolved = tf.layers.conv1d(
            tf.reshape(droppedout, [-1, channels_in, self._window]),
            filters=channels_out,
            kernel_size=kernel_length,
            strides=1,
            padding='same',
            data_format='channels_first',
            activation=self._activation,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer()
        ) # (batchsize, channels_out, window)
        
        pooled, ind = tf.nn.max_pool_with_argmax(
            tf.reshape(convolved, [-1, channels_out, self._window, 1]),
            ksize=[1,1,2,1],
            strides=[1,1,2,1],
            padding="VALID"
        ) # (batchsize, channels_out, window//2, 1)
        
        self._pooling_ind_stack.append(ind)
        
        manifold = tf.reshape(pooled, [-1, channels_out, self._window//2])
        return manifold

    def conv_out(self, x, channels_in, channels_out, kernel_length, dropout=0.25):
        if self._accurate_depooling:
            depooled = self.unpool(
                tf.reshape(x, [-1, channels_in, self._window//2, 1]),
                self._pooling_ind_stack.pop(),
                stride=[1, 1, 2, 1]
            ) # (batchsize, channels_in, window, 1)
        else:
            depooled = tf.multiply(tf.image.resize_images(
                tf.reshape(x, [-1, channels_in, self._window//2, 1]),
                [channels_in, self._window],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            ), 0.5) # (batchsize, channels_in, window, 1)

        droppedout = tf.layers.dropout(
            depooled,
            rate=dropout,
            seed=None,
            training=self.training
        )

        deconvolved = tf.layers.conv1d(
            tf.reshape(droppedout, [-1, channels_in, self._window]),
            filters=channels_out,
            kernel_size=kernel_length,
            strides=1,
            padding='same',
            data_format='channels_first',
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer()
        ) # (batchsize, channels_out, window)
        
        return deconvolved
    
    @staticmethod
    def unpool(pool, ind, stride=[1, 2, 2, 1], scope='unpool_2d'):
      """Adds a 2D unpooling op.
      https://arxiv.org/abs/1505.04366
      Unpooling layer after max_pool_with_argmax.
           Args:
               pool:        max pooled output tensor
               ind:         argmax indices
               stride:      stride is the same as for the pool
           Return:
               unpool:    unpooling tensor
      """
      with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]
    
        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]
    
        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                          shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)
    
        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)
    
        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret
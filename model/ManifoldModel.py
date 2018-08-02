import tensorflow as tf

class ManifoldModel(object):
    def __init__(self, 
                 window,
                 activation=tf.nn.relu,
                 maxpooling=True,
                 accurate_depooling=False,
                 batch_norm=True,
                 residual=False):
        self._window = window
        self._depth = 1
        self._activation = activation
        self._pooling_ind_stack = []
        self._accurate_depooling = accurate_depooling
        self._maxpooling = maxpooling
        self._batch_norm = batch_norm
        self._residual = residual
        
        self.training = tf.placeholder(tf.bool, [])
    def conv_in(self, x, channels_in, channels_out, kernel_length, dropout=0.25, kernel_weights=None, bias_weights=None):
        droppedout = tf.layers.dropout(
            x,
            rate=dropout,
            seed=None,
            training=self.training
        )
        
        convolved = tf.layers.conv1d(
            tf.reshape(droppedout, [-1, channels_in, self._window//self._depth]),
            filters=channels_out,
            kernel_size=kernel_length,
            strides=1 if self._maxpooling else 2,
            padding='same',
            data_format='channels_first',
            activation=None,
            kernel_initializer=tf.constant_initializer(kernel_weights) if kernel_weights is not None else None,
            bias_initializer=tf.constant_initializer(bias_weights) if bias_weights is not None else tf.zeros_initializer(),
            trainable=False if kernel_weights is not None and bias_weights is not None else True
        ) # (batchsize, channels_out, window)
        
        if self._batch_norm:
            convolved = tf.layers.batch_normalization(convolved, axis=1, momentum=0.99995, training=self.training, fused=True)
        
        if self._activation:
            convolved = self._activation(convolved)
        
        if self._maxpooling:
            pooled, ind = tf.nn.max_pool_with_argmax(
                tf.reshape(convolved, [-1, channels_out, self._window//self._depth, 1]),
                ksize=[1,1,2,1],
                strides=[1,1,2,1],
                padding="VALID"
            ) # (batchsize, channels_out, window//2, 1)
            self._pooling_ind_stack.append(ind)
            manifold = tf.reshape(pooled, [-1, channels_out, self._window//(self._depth*2)])
            
        else:
            manifold = tf.reshape(convolved, [-1, channels_out, self._window//(self._depth*2)])
            
        self._depth *= 2
        return manifold

    def conv_out(self, x, channels_in, channels_out, kernel_length, dropout=0.25, residual_candidate=None, output_layer=False, kernel_weights=None, bias_weights=None):
        self._depth //= 2
        
        if self._maxpooling and self._accurate_depooling:
            depooled = self.unpool(
                tf.reshape(x, [-1, channels_in, self._window//(self._depth*2), 1]),
                self._pooling_ind_stack.pop(),
                stride=[1, 1, 2, 1]
            ) # (batchsize, channels_in, window, 1)
        else:
            depooled = tf.multiply(tf.image.resize_images(
                tf.reshape(x, [-1, channels_in, self._window//(self._depth*2), 1]),
                [channels_in, self._window//self._depth],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            ), 0.5) # (batchsize, channels_in, window, 1)

        droppedout = tf.layers.dropout(
            depooled,
            rate=dropout,
            seed=None,
            training=self.training
        )

        deconvolved = tf.layers.conv1d(
            tf.reshape(droppedout, [-1, channels_in, self._window//self._depth]),
            filters=channels_out,
            kernel_size=kernel_length,
            strides=1,
            padding='same',
            data_format='channels_first',
            activation=None,
            kernel_initializer=tf.constant_initializer(kernel_weights) if kernel_weights is not None else None,
            bias_initializer=tf.constant_initializer(bias_weights) if bias_weights is not None else tf.zeros_initializer(),
            trainable=False if kernel_weights is not None and bias_weights is not None else True
        ) # (batchsize, channels_out, window)
        
        if self._residual and residual_candidate is not None:
            deconvolved = deconvolved + residual_candidate
        
        if (not output_layer) and self._batch_norm:
            deconvolved = tf.layers.batch_normalization(deconvolved, axis=1, momentum=0.99995, training=self.training, fused=True)
        
        if not output_layer and self._activation:
            deconvolved = self._activation(deconvolved)
        
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

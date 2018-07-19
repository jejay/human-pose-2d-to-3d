from ManifoldModel import ManifoldModel
import tensorflow as tf

class FlexibleSerialManifoldModel(ManifoldModel):
    def __init__(self,
                 window,
                 data_channels_in,
                 data_channels_out,
                 manifold_depth=1,
                 num_manifolds=1,
                 filter_length=25,
                 conv_channels=256,
                 dropout=0.25,
                 batch_norm=True,
                 intra_residual=True,
                 inter_residual=True,
                 maxpooling=True,
                 accurate_depooling=False,
                 first_layer_dropout=False):
        super().__init__(window,
                         batch_norm=batch_norm,
                         residual=True,
                         maxpooling=maxpooling,
                         accurate_depooling=accurate_depooling)
        self._data_channels_in = data_channels_in
        self._data_channels_out = data_channels_out
        self._manifold_depth = manifold_depth
        self._num_manifolds = num_manifolds
        self._filter_length = filter_length
        self._conv_channels = conv_channels
        self._dropout = dropout
        self._intra_residual = intra_residual
        self._inter_residual = inter_residual
        self._first_layer_dropout = first_layer_dropout
    
    def build_graph(self, inputs):
        x = inputs
        residual_candidates = []
        self.manifolds = []
        self.autoencoded = []
        
        with tf.variable_scope("serial_manifold_{0}x{1}".format(self._filter_length, self._window)):
            for i in range(self._num_manifolds):
                with tf.variable_scope("manifold_{0}".format(i)):
                    residual_candidates.append(None if not self._inter_residual else x)
                    channels_out = self._data_channels_in if i == 0 else self._data_channels_out
                    
                    for j in range(self._manifold_depth):
                        channels_in = channels_out
                        channels_out = int(self._conv_channels * (2**j))
                        with tf.variable_scope("conv_{0}_to_{1}".format(channels_in, channels_out)):    
                            x = self.conv_in(x,
                                             channels_in=channels_in,
                                             channels_out=channels_out,
                                             kernel_length=self._filter_length,
                                             dropout=self._dropout if j > 0 or i > 0 or self._first_layer_dropout else 0)
                            residual_candidates.append(None if not self._intra_residual else x)
                    
                    
                    self.manifolds.append(x)
                    residual_candidates.pop() # The deepest conv can never be used for residual connections
                        
                    for j in range(self._manifold_depth):
                        channels_in = channels_out
                        channels_out = int(channels_out//2) if j < self._manifold_depth-1 else self._data_channels_out
                        with tf.variable_scope("deconv_{0}_to_{1}".format(channels_in, channels_out)):   
                            x = self.conv_out(x,
                                              channels_in=channels_in,
                                              channels_out=channels_out,
                                              kernel_length=self._filter_length,
                                              dropout=self._dropout,
                                              residual_candidate=None if len(residual_candidates) == 0 else residual_candidates.pop(),
                                              output_layer=True if i == self._num_manifolds-1 and j == self._manifold_depth-1 else False)
                    self.autoencoded.append(x)

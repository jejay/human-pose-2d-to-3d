from ManifoldModel import ManifoldModel
import tensorflow as tf

class DeepManifoldModel(ManifoldModel):
    def __init__(self, window, batch_norm=True, residual=True):
        super().__init__(window, batch_norm=batch_norm, residual=residual)
    
    def build_graph(self, inputs):
        with tf.variable_scope("deep_manifold"):
            self.inputs = inputs
           
            with tf.variable_scope("first_conv"): 
                self.first_conv = self.conv_in(inputs,
                                               channels_in= 77,
                                               channels_out=128,
                                               kernel_length=21,
                                               dropout=0)
            with tf.variable_scope("second_conv"): 
                self.second_conv = self.conv_in(self.first_conv,
                                                channels_in=128,
                                                channels_out=256,
                                                kernel_length=21,
                                                dropout=0.25)
            with tf.variable_scope("manifold"): 
                self.manifold = self.conv_in(self.second_conv ,
                                             channels_in=256,
                                             channels_out=512,
                                             kernel_length=21,
                                             dropout=0.25)
            
            with tf.variable_scope("first_deconv"): 
                self.first_deconv = self.conv_out(self.manifold,
                                                  channels_in=512,
                                                  channels_out=256,
                                                  kernel_length=21,
                                                  dropout=0.25,
                                                  residual_candidate=self.second_conv)
            with tf.variable_scope("second_deconv"): 
                self.second_deconv = self.conv_out(self.first_deconv,
                                                   channels_in=256,
                                                   channels_out=128,
                                                   kernel_length=21,
                                                   dropout=0.25,
                                                   residual_candidate=self.first_conv)
            with tf.variable_scope("autoencoded"): 
                self.autoencoded =  self.conv_out(self.second_deconv,
                                                  channels_in=128,
                                                  channels_out= 77,
                                                  kernel_length=21,
                                                  dropout=0.25,
                                                  output_layer=True)
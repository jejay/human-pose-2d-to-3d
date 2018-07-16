from ManifoldModel import ManifoldModel
import tensorflow as tf

class SingleManifoldModel(ManifoldModel):
    def __init__(self, window, activation=tf.nn.relu, maxpooling=True, accurate_depooling=False):
        super().__init__(window, activation=activation, maxpooling=maxpooling, accurate_depooling=accurate_depooling)
    
    def build_graph(self, inputs):
        with tf.variable_scope("single_layer_manifold"):
            self.manifold = self.conv_in(inputs, channels_in=73, channels_out=256, kernel_length=25)
            self.autoencoded = self.conv_out(self.manifold, channels_in=256, channels_out=73, kernel_length=25, output_layer=True)
        

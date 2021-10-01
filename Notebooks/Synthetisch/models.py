"""
Import Libraries
"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

class Dense(layers.Layer):
    """
    Create the dense layer (only inputs with weights and bias)
    """
    def __init__(self, units):
        
        super(Dense, self).__init__()
        self.units = units
    def build(self, input_shape):
        self.W = self.add_weight(
            name = 'W',
            shape = (input_shape[-1], self.units),
            initializer = 'random_normal',
            trainable = True,
        )
        self.b = self.add_weight(
            name='b',
            shape = (self.units, ), initializer = "zeros", trainable = True,
        )
    def call(self, inputs):
        """
        Does the matrix multiplication from input and weights + biases 
        """
        print()
        M_a = tf.matmul(inputs, self.W) + self.b # initial Matrix
        return M_a

class MyReLU(layers.Layer):
    """
    Manually build the activation function (ReLU)
    """
    def __init__(self):
        super(MyReLU, self).__init__()
    def call(self, x):
        return tf.math.maximum(x, 0)

class MyLeakyReLU(layers.Layer):
    """
    Manually build the activation function (LeakyReLU)
    """
    def __init__(self):
        super(MyReLU, self).__init__()
    def call(self, x, alpha):
        if x >= 0:
            return x
        else:
            return x * alpha
        
class Encoder(layers.Layer):
    """
    Creating a two layers encoder
    """
    def __init__(self, l
                 atent_dim = 50, 
                 intermediate_dim = 200, 
                 name="Encoder", 
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense1 = Dense(intermediate_dim)
        self.dense2 = Dense(latent_dim)
        self.relu = MyReLU()
    def call(self, input_tensor):
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)
    
class Decoder(layers.Layer):
    """
    Creating a two layer Decoder
    """
    def __init__(self, original_dim, intermediate_dim=64, name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense3 = Dense(intermediate_dim)
        self.dense4 = Dense(original_dim)
        self.relu = MyReLU()
    def call(self, input_tensor):
        x = self.relu(self.dense3(input_tensor))
        return self.dense4(x)
    
    
class AutoEncoder(keras.Model):
    """
    Combining Encoder and Decoder
    """
    def __init__(
        self,
        original_dim,
        intermediate_dim = 100,
        latent_dim = 46,
        name="autoencoder",
        **kwargs
    ):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim = latent_dim, intermediate_dim = intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim = intermediate_dim)
    def call(self, inputs):
        z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed
    def model(self, input_len):
        x = keras.Input(shape = input_len)
        print(x)
        return keras.Model(inputs = [x], outputs = self.call(x))
    
class Abyss_Encoder(keras.Model):
    """
    
    """
    def __init__(
        self,
        original_dim,
        intermediate_dim = 4096,
        latent_dim = 46,
        name="encoder",
        **kwargs
    ):
        super(Abyss_Encoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim = latent_dim, intermediate_dim = intermediate_dim)
    def call(self, inputs):
        z = self.encoder(inputs)
        return z
    def model(self, input_len):
        x = keras.Input(shape = input_len)
        print(x)
        return keras.Model(inputs = [x], outputs = self.call(x))
    
class FrequencyFinder(keras.Model):
    """
    crazy ass model finding the frequency through random sampling
    """
    def __init__(
        self,
        population_layer = 10,
        name = "frequency finder",
        **kwargs
    ):
        super(FrequencyFinder, self).__init__(name = name, **kwargs)
        self.population_layer = population_layer
        self.link = 
    def call(self, inputs):
        z = 
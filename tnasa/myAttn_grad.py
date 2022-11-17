import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
random.seed(10)
import time
import math
import typing
import warnings

class mha(tf.keras.layers.Layer):

    def __init__(
            self,
            head_size: int,
            num_heads: int,
            output_size: int = None,
            dropout: float = 0.0,
            use_projection_bias: bool = True,
            return_attn_coef: bool = True,
            kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
            kernel_regularizer: typing.Union[str, typing.Callable] = None,
            kernel_constraint: typing.Union[str, typing.Callable] = None,
            bias_initializer: typing.Union[str, typing.Callable] = "zeros",
            bias_regularizer: typing.Union[str, typing.Callable] = None,
            bias_constraint: typing.Union[str, typing.Callable] = None,

            **kwargs,
    ):

        super().__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self._droput_rate = dropout


    def build(self, input_shape):

        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        query = inputs[0]
        key = inputs[1]
        value = inputs[2] if len(inputs) > 2 else key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )
        """
        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )
        """
        # Linear transformations
        query = tf.einsum("...NI , HIO -> ...NHO", query, self.query_kernel)
        key = tf.einsum("...MI , HIO -> ...MHO", key, self.key_kernel)
        value = tf.einsum("...MI , HIO -> ...MHO", value, self.value_kernel)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=query.dtype)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        """
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)
        """
        attn_coef = tf.nn.softmax(logits)

            # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef, value 
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        else:
            return output_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout=self._droput_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
            kernel_initializer=tf.keras.initializers.serialize(self.kernel_initializer),
            kernel_regularizer=tf.keras.regularizers.serialize(self.kernel_regularizer),
            kernel_constraint=tf.keras.constraints.serialize(self.kernel_constraint),
            bias_initializer=tf.keras.initializers.serialize(self.bias_initializer),
            bias_regularizer=tf.keras.regularizers.serialize(self.bias_regularizer),
            bias_constraint=tf.keras.constraints.serialize(self.bias_constraint),

        )

        return config

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        self.num_hiddens=num_hiddens
        self.dropout=dropout
        self.max_len=max_len
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(self.dropout)
        # Create a long enough `P`
        self.P = np.zeros((1, self.max_len, self.num_hiddens))
        X_p = np.arange(self.max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, self.num_hiddens, 2, dtype=np.float32) / self.num_hiddens)
        self.P[:, :, 0::2] = np.sin(X_p)
        self.P[:, :, 1::2] = np.cos(X_p)

    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_hiddens": self.num_hiddens,
            "dropout": self.dropout,
            "max_len": self.max_len,        
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AttentionBlock(layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_dim,
                 rate=0.1,
                 return_attn_coef=True,
                 layer_or_batch="batch",
                 **kwargs):


        super(AttentionBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate  # dropoff rate
        self.return_attn_coef = return_attn_coef
        self.layer_or_batch=layer_or_batch

        self.att = mha(head_size=int(embed_dim / num_heads),
                                            num_heads=self.num_heads,
                                            return_attn_coef=self.return_attn_coef
                                            )
        self.ffn = keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim), ]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)
        
        self.batchnorm1= layers.BatchNormalization()
        self.batchnorm2= layers.BatchNormalization()

    def call(self, inputs, training, **kwargs):
        if self.return_attn_coef:

            attn_output, attn_scores, myvalue = self.att([inputs, inputs])

        else:
            attn_output = self.att([inputs, inputs])
            
        attn_output = self.dropout1(attn_output, training=training)
        
        if self.layer_or_batch =="batch":
            out1 = self.batchnorm1(inputs + attn_output, training=training)
        elif self.layer_or_batch =="layer":
            out1 = self.layernorm1(inputs + attn_output)
            
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        if self.layer_or_batch =="batch":
            attn_out_last = self.batchnorm2(out1 + ffn_output, training=training)
        elif self.layer_or_batch =="layer":
            attn_out_last = self.layernorm2(out1 + ffn_output)
        
        if self.return_attn_coef:

            return attn_out_last, attn_scores,myvalue 
        else:
            return attn_out_last

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'batchnorm1': self.batchnorm1,
            'batchnorm2': self.batchnorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'return_attn_coef': self.return_attn_coef,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'layer_or_batch':self.layer_or_batch
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
####################### Jul 26 add maxpool
class myMaxPool(tf.keras.layers.Layer):
    """This max pool function takes the max of forward and rc 
        of motif scores
        
        Input dim: (sample n (n), seq len (l), # of motif lib (m))
        step 1: expand to (n,l,m,1)
        step 2:  maxpool of 1,2 step, that don't affect the l, 
            while taking max of every two (forward and rc)
        step 3: (n,l,m/2,1) --> (n,l,m/2)
    """
    def __init__(self ):
        super().__init__()
        self.maxpool2d = tf.keras.layers.MaxPool2D(pool_size=(1, 2)) 


    def call(self, X, **kwargs):
        X = tf.expand_dims(X,axis= -1)
        X = self.maxpool2d(X)
        X = tf.squeeze(X, axis= -1)
        return X
        
    
    def get_config(self):
        config = super().get_config()
        config.update({})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

########################## Aug 17 add layer for attn dot value 
class AttentionGrad_preLayer(layers.Layer):
    def __init__(self,
                 value_shape,
                 **kwargs):
        super().__init__()
        self.value_shape = value_shape
        self.myv= tf.zeros(self.value_shape)
    def call(self, inputs, training, **kwargs):
        multihead_product= tf.einsum("...HNM,...MHI->...NHI", inputs, self.myv)
        return multihead_product


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            #'myvalue': self.myvalue,
            'value_shape': self.value_shape,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



########################## Aug 16 add attention grad tracing 
class AttentionGrad(layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_dim,
                 pre_input_dim,
                 mykernel,
                 mybias,
                 layer_or_batch="batch",
                 **kwargs):


        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.layer_or_batch=layer_or_batch
        self.pre_input_dim=pre_input_dim
        self.mykernel=mykernel
        self.mybias=mybias

        self.ffn = keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim), ]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        
        self.batchnorm1= layers.BatchNormalization()
        self.batchnorm2= layers.BatchNormalization()

        self.pre_input = tf.zeros(pre_input_dim)

    def call(self, inputs, training, **kwargs):
        #multihead_output = tf.einsum("...HNM,...MHI->...NHI", inputs, self.myvalue)
        #input is multihead_output, that is attn dot value
        output_mha = tf.einsum("...NHI,HIO->...NO", inputs, self.mykernel )
        output_mha += self.mybias

        if self.layer_or_batch =="batch":
            out1 = self.batchnorm1(self.pre_input + output_mha, training=training)
        elif self.layer_or_batch =="layer":
            out1 = self.layernorm1(self.pre_input + output_mha)
            
        ffn_output = self.ffn(out1)
        
        if self.layer_or_batch =="batch":
            attn_out_last = self.batchnorm2(out1 + ffn_output, training=training)
        elif self.layer_or_batch =="layer":
            attn_out_last = self.layernorm2(out1 + ffn_output)
        

        return attn_out_last

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pre_input_dim': self.pre_input_dim,
            'pre_input': self.pre_input,
            'mykernel': self.mykernel,
            'mybias': self.mybias,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'batchnorm1': self.batchnorm1,
            'batchnorm2': self.batchnorm2,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'layer_or_batch':self.layer_or_batch
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
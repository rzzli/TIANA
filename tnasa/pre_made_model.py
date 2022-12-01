import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
tf.random.set_seed(1)
import time
import math
import sys
import random
from tensorflow.keras.models import load_model
print(sys.path)
sys.path.append('/gpfs/data01/glasslab/home/zhl022/daima/to_share/DeepLearningAttention/round2_code')
from model_layers import *
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling1D, MaxPooling1D
import tensorflow.keras.backend as K


def make_model_attn( output_bias=None, lr=5e-5):
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      tf.keras.metrics.Recall(name='recall_5',thresholds=0.5),
      tf.keras.metrics.Recall(name='recall_1',thresholds=0.1)
    ]
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    batch_size = 128
    num_classes = 1
    seq_size = 300
    n_channels = 4
    input_shape = (seq_size, n_channels)
    dropout_rate=0.4
    filter_size=22
    filter_stride=1
    layer1_out_length=seq_size-(filter_size/filter_stride)+1
    layer1_out_maxpool=math.ceil(layer1_out_length*1.0/4)
    first_filter_num=560
    optimizer_learning=tf.keras.optimizers.Adam(learning_rate=lr)

    inputs = layers.Input(shape=input_shape)
    x=layers.Conv1D(first_filter_num, kernel_size=filter_size, strides=1, padding='valid',
        kernel_initializer=tf.constant_initializer(1e-6),
        use_bias=False,
        input_shape=input_shape, name='conv1')(inputs)
    mymax=myMaxPool()
    x=mymax(x)
    x=layers.BatchNormalization(name = "bn1")(x)
    x=layers.Activation('relu')(x)
    x=layers.MaxPooling1D(pool_size=4, strides=4, padding='same')(x)
    x=layers.Dropout(dropout_rate)(x)

    pos_encoding= PositionalEncoding(first_filter_num//2, 0, int(layer1_out_maxpool))
    x=pos_encoding(x)

    transformer1 = AttentionBlock(first_filter_num//2, 4, first_filter_num*2//2, return_attn_coef=True,rate=dropout_rate,layer_or_batch="batch")
    x,weights,_ = transformer1(x)

    x = layers.Flatten()(x)
    x=layers.Dropout(dropout_rate)(x)
    x=layers.Dense(256,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8))(x)
    x=layers.Activation('relu')(x)
    x=layers.Dropout(dropout_rate)(x)
    x=layers.Dense(64,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8))(x)
    x=layers.Activation('relu')(x)
    outputs=layers.Dense(1, activation='sigmoid',bias_initializer=output_bias)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    with open('/home/zhl022/daima/projects/dl_methods_improve/motif_pssm_Aug27/motif_npy/motif530_rc_pssm.npy', 'rb') as f:
        padded_motif_np_swap= np.load(f)
    weights=model.get_weights()
    weights[0][...,:530]=padded_motif_np_swap
    model.set_weights(weights)
    model.layers[1].trainable=False
    model.compile(optimizer=optimizer_learning, loss="binary_crossentropy", metrics=METRICS)
    return model




## deep sea
def make_model_deepsea(  output_bias=None,lr=5e-5):
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      tf.keras.metrics.Recall(name='recall_5',thresholds=0.5),
      tf.keras.metrics.Recall(name='recall_1',thresholds=0.1)
    ]
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    batch_size = 128
    num_classes = 1
    seq_size = 300
    n_channels = 4
    input_shape = (seq_size, n_channels)
    # model_il4 structure
    model = Sequential()
    # conv1
    model.add(Conv1D(320, kernel_size=8, strides=1, padding='valid',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8),
        input_shape=input_shape, name='conv1'))
    model.add(BatchNormalization(name = "bn1"))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='same'))
    model.add(Dropout(0.2))

    # conv2
    model.add(Conv1D(480, kernel_size=8, strides=1, padding='valid',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8),
        name='conv2'))
    model.add(BatchNormalization(name = "bn2"))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='same'))
    model.add(Dropout(0.2))

    # conv3
    model.add(Conv1D(960, kernel_size=8, strides=1, padding='valid',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8),
        name='conv3'))
    model.add(BatchNormalization(name = "bn3"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    # fc1
    model.add(Dense(925,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8)))
    model.add(Activation('relu'))

    # output
    model.add(Dense(num_classes,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=output_bias,
        activity_regularizer=keras.regularizers.l1(1e-8)))
    model.add(Activation('sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=METRICS)
    #"""
    def load_from_tf(name):
        return tf.train.load_variable("/home/zes017/AD/data/AgentBind/benchmarking_dnase/data/model", name=name)
    model.get_layer("conv1").set_weights(
        [load_from_tf('conv1/weights/ExponentialMovingAverage'),
        load_from_tf('conv1/biases/ExponentialMovingAverage')])
    model.get_layer("conv2").set_weights(
        [load_from_tf('conv2/weights/ExponentialMovingAverage'),
        load_from_tf('conv2/biases/ExponentialMovingAverage')])
    model.get_layer("conv3").set_weights(
        [load_from_tf('conv3/weights/ExponentialMovingAverage'),
        load_from_tf('conv3/biases/ExponentialMovingAverage')])
    model.get_layer("bn1").set_weights(
        [load_from_tf('conv1/batch_normalization/gamma/ExponentialMovingAverage'),
         load_from_tf('conv1/batch_normalization/beta/ExponentialMovingAverage'),
        np.zeros(model.get_layer("bn1").weights[2].shape),
        np.zeros(model.get_layer("bn1").weights[3].shape)])
    model.get_layer("bn2").set_weights(
        [load_from_tf('conv2/batch_normalization/gamma/ExponentialMovingAverage'),
         load_from_tf('conv2/batch_normalization/beta/ExponentialMovingAverage'),
        np.zeros(model.get_layer("bn2").weights[2].shape),
        np.zeros(model.get_layer("bn2").weights[3].shape)])
    model.get_layer("bn3").set_weights(
        [load_from_tf('conv3/batch_normalization/gamma/ExponentialMovingAverage'),
         load_from_tf('conv3/batch_normalization/beta/ExponentialMovingAverage'),
        np.zeros(model.get_layer("bn3").weights[2].shape),
        np.zeros(model.get_layer("bn3").weights[3].shape)])
    return model







def make_model_deepstarr_short( output_bias=None,lr=5e-5):
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      tf.keras.metrics.Recall(name='recall_5',thresholds=0.5),
      tf.keras.metrics.Recall(name='recall_1',thresholds=0.1)
    ]
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = Sequential()
    model.add(Conv2D(16, (12, 4), padding='valid', input_shape=(300, 4, 1),
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=(15,1), padding='valid'))

    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation='sigmoid' ,bias_initializer=output_bias))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='sgd',metrics=METRICS)
    return model
    

def make_model_deepstarr_long(output_bias=None,lr=5e-5):
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      tf.keras.metrics.Recall(name='recall_5',thresholds=0.5),
      tf.keras.metrics.Recall(name='recall_1',thresholds=0.1)
    ]
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    params = {'batch_size': 128,
          'epochs': 100,
          'early_stop': 10,
          'kernel_size1': 7,
          'kernel_size2': 3,
          'kernel_size3': 5,
          'kernel_size4': 3,
          'lr': lr,
          'num_filters': 256,
          'num_filters2': 60,
          'num_filters3': 60,
          'num_filters4': 120,
          'n_conv_layer': 4,
          'n_add_layer': 2,
          'dropout_prob': 0.4,
          'dense_neurons1': 256,
          'dense_neurons2': 256,
          'pad':'same'}
    
    lr = params['lr']
    dropout_prob = params['dropout_prob']
    n_conv_layer = params['n_conv_layer']
    n_add_layer = params['n_add_layer']
    
    # body
    input = layers.Input(shape=(300, 4))
    x = layers.Conv1D(params['num_filters'], kernel_size=params['kernel_size1'],
                  padding=params['pad'],
                  name='Conv1D_1st')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    for i in range(1, n_conv_layer):
        x = layers.Conv1D(params['num_filters'+str(i+1)],
                      kernel_size=params['kernel_size'+str(i+1)],
                      padding=params['pad'],
                      name=str('Conv1D_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
    
    x = Flatten()(x)
    
    # dense layers
    for i in range(0, n_add_layer):
        x = layers.Dense(params['dense_neurons'+str(i+1)],
                     name=str('Dense_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_prob)(x)
    outputs=layers.Dense(1, activation='sigmoid',
                             bias_initializer=output_bias)(x)

    model = keras.models.Model([input], outputs)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy", metrics=METRICS)
    
    return model

####### OCT 2 use abtba motif
def make_model_attn_abtba( output_bias=None, lr=5e-5):
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      tf.keras.metrics.Recall(name='recall_5',thresholds=0.5),
      tf.keras.metrics.Recall(name='recall_1',thresholds=0.1)
    ]
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    batch_size = 128
    num_classes = 1
    seq_size = 300
    n_channels = 4
    input_shape = (seq_size, n_channels)
    dropout_rate=0.4
    filter_size=20
    filter_stride=1
    layer1_out_length=seq_size-(filter_size/filter_stride)+1
    layer1_out_maxpool=math.ceil(layer1_out_length*1.0/4)
    first_filter_num=520
    optimizer_learning=tf.keras.optimizers.Adam(learning_rate=lr)

    inputs = layers.Input(shape=input_shape)
    x=layers.Conv1D(first_filter_num, kernel_size=filter_size, strides=1, padding='valid',
        kernel_initializer=tf.constant_initializer(1e-6),
        use_bias=False,
        input_shape=input_shape, name='conv1')(inputs)
    mymax=myMaxPool()
    x=mymax(x)
    x=layers.BatchNormalization(name = "bn1")(x)
    x=layers.Activation('relu')(x)
    x=layers.MaxPooling1D(pool_size=4, strides=4, padding='same')(x)
    x=layers.Dropout(dropout_rate)(x)

    pos_encoding= PositionalEncoding(first_filter_num//2, 0, int(layer1_out_maxpool))
    x=pos_encoding(x)

    transformer1 = AttentionBlock(first_filter_num//2, 4, first_filter_num*2//2, return_attn_coef=True,rate=dropout_rate,layer_or_batch="batch")
    x,weights,_ = transformer1(x)

    x = layers.Flatten()(x)
    x=layers.Dropout(dropout_rate)(x)
    x=layers.Dense(256,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8))(x)
    x=layers.Activation('relu')(x)
    x=layers.Dropout(dropout_rate)(x)
    x=layers.Dense(64,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8))(x)
    x=layers.Activation('relu')(x)
    outputs=layers.Dense(1, activation='sigmoid',bias_initializer=output_bias)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    with open('/home/zhl022/daima/projects/dl_jenhan_motif/make_motif/motif518_rc_pssm.npy', 'rb') as f:
        padded_motif_np_swap= np.load(f)
    weights=model.get_weights()
    weights[0][...,:518]=padded_motif_np_swap
    model.set_weights(weights)
    model.layers[1].trainable=False
    model.compile(optimizer=optimizer_learning, loss="binary_crossentropy", metrics=METRICS)
    return model


####### OCT 3 use abtba motif
def make_model_attn_cluster(num_tf,pad_size,max_len,pssm_path, output_bias=None, lr=5e-5):
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      tf.keras.metrics.Recall(name='recall_5',thresholds=0.5),
      tf.keras.metrics.Recall(name='recall_1',thresholds=0.1)
    ]
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    batch_size = 128
    num_classes = 1
    seq_size = 300
    n_channels = 4
    input_shape = (seq_size, n_channels)
    dropout_rate=0.4
    filter_size=max_len
    filter_stride=1
    layer1_out_length=seq_size-(filter_size/filter_stride)+1
    layer1_out_maxpool=math.ceil(layer1_out_length*1.0/4)
    first_filter_num=num_tf*2 + pad_size*2
    optimizer_learning=tf.keras.optimizers.Adam(learning_rate=lr)

    inputs = layers.Input(shape=input_shape)
    x=layers.Conv1D(first_filter_num, kernel_size=filter_size, strides=1, padding='valid',
        kernel_initializer=tf.constant_initializer(1e-6),
        use_bias=False,
        input_shape=input_shape, name='conv1')(inputs)
    mymax=myMaxPool()
    x=mymax(x)
    x=layers.BatchNormalization(name = "bn1")(x)
    x=layers.Activation('relu')(x)
    x=layers.MaxPooling1D(pool_size=4, strides=4, padding='same')(x)
    x=layers.Dropout(dropout_rate)(x)

    pos_encoding= PositionalEncoding(first_filter_num//2, 0, int(layer1_out_maxpool))
    x=pos_encoding(x)

    transformer1 = AttentionBlock(first_filter_num//2, 4, first_filter_num*2//2, return_attn_coef=True,rate=dropout_rate,layer_or_batch="batch")
    x,weights,_ = transformer1(x)

    x = layers.Flatten()(x)
    x=layers.Dropout(dropout_rate)(x)
    x=layers.Dense(256,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8))(x)
    x=layers.Activation('relu')(x)
    x=layers.Dropout(dropout_rate)(x)
    x=layers.Dense(64,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
        kernel_regularizer=keras.regularizers.l2(5e-7),
        bias_initializer=keras.initializers.Constant(value=0),
        activity_regularizer=keras.regularizers.l1(1e-8))(x)
    x=layers.Activation('relu')(x)
    outputs=layers.Dense(1, activation='sigmoid',bias_initializer=output_bias)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    with open(pssm_path, 'rb') as f:
        padded_motif_np_swap= np.load(f)
    weights=model.get_weights()
    weights[0][...,:num_tf*2]=padded_motif_np_swap
    model.set_weights(weights)
    model.layers[1].trainable=False
    model.compile(optimizer=optimizer_learning, loss="binary_crossentropy", metrics=METRICS)
    return model








































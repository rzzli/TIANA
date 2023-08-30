import numpy as np
import pandas as pd
from random import choices
from random import sample
from itertools import combinations
from scipy import stats
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(1)
import sys
#from layers_helper import PositionalEncoding,AttentionBlock
sys.path.append('/gpfs/data01/glasslab/home/zhl022/daima/to_share/DeepLearningAttention/round3_code')
from model_layers import *
from pre_made_model import *
from ig_attn import *
from attn_weights import *
from sklearn.model_selection import train_test_split

tf.config.run_functions_eagerly(True)

### maybe not necessary imports
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tensorflow.keras.models import load_model
from statannot import add_stat_annotation
from scipy.stats import rankdata
import imblearn
from imblearn.under_sampling import RandomUnderSampler 

def data_loader(pos_train_path,pos_val_path,neg_path):
    """
    Load data from .npy files to one-hot encodings
    Args (required):
        pos_train_path
        pos_val_path
        neg_path
    #'/home/zhl022/daima/projects/make_peaks_Aug30/ldtf_maggie/processed_peaks/PU1/C57_PU1_spec_BALB_ref_train.npy'
    #'/home/zhl022/daima/projects/make_peaks_Aug30/ldtf_maggie/processed_peaks/PU1/C57_PU1_spec_BALB_ref_val.npy'
    #"/home/zhl022/daima/projects/make_peaks_Aug30/negative_peaks/mm10/npy/mm10_neg200bp_200k_2.npy"
    Returns:
        X_pos_train,X_pos_val,X_neg_train,X_neg_val
    """
    with open(pos_train_path, 'rb') as f:
        X_pos_train= np.load(f)
    with open(pos_val_path, 'rb') as f:
        X_pos_val= np.load(f)
    with open(neg_path, 'rb') as f:
        X_neg = np.load(f)
    ### split negs into validation and testing set so that avoid overlap
    Y_neg = np.zeros(X_neg.shape[0])
    X_neg_train, X_neg_val, _ , _ = train_test_split(X_neg, Y_neg, test_size=0.2, random_state=42) # do not need Y here
    return X_pos_train,X_pos_val,X_neg_train,X_neg_val
         
    
def trainSub(Xpos,Xneg,random_state=42):
    """
    Subset training class to balance class and return training data mixed of positive and negative class
    Args:
        Xpos: positive dataset (smaller than neg)
        Xnog: negative dataset (larget than pos)
        random_state (optional): seed
    Returns:
        X and Y for training, with matched mixed positive and negative classes
    """
    random.seed(10)
    random_idx_neg=random.sample(range(Xneg.shape[0]),Xpos.shape[0] )
    Xneg_sub = Xneg[random_idx_neg,...]
    ypos=np.ones(Xpos.shape[0])
    yneg=np.zeros(Xneg_sub.shape[0]) 
    
    X_train=np.concatenate([Xpos,Xneg_sub],axis=0)
    y_train=np.concatenate([ypos,yneg],axis=-1)

    return X_train, y_train


def check_consecutive_difference(loss_values,window_size=5,thres=0.008,wait=3,min_epoch=15):
    """
    Check if the loss value (list) stops improving (or converged ).
    Args:
        loss_values: list of losses, either prc or auc
        window_size: size of window to average the past loss
        thres: float point cutoff
        wait: patience number, if the change in loss scores is less than thres for 3(or other specified cycles), return true
    Return:
        False: if has not converged OR less than window size
        True: if converged
    """
    if len(loss_values) < window_size:
        return False
    
    if len(loss_values) < min_epoch:
        return False
    consecutive_count = 0
    moving_averages = []
    for i, loss in enumerate(loss_values):
        if i >= window_size - 1:
            moving_avg = np.mean(loss_values[i - window_size + 1 : i + 1])
            moving_averages.append(moving_avg)
            
            if i >= window_size + 1:
                diff = loss - moving_averages[-1]
                if abs(diff) < thres:
                    consecutive_count += 1
                    if consecutive_count >= wait:
                        return True
                else:
                    consecutive_count = 0
    return False

def train(pssm,
          pos_train_path,
          pos_val_path,
          neg_path,
          learning_rate = 1e-5,
          nepoch=50,
          early_stop=True,
          early_method='auc',
          early_window=5,
          early_wait=3,
          early_thres=0.008,
          min_epoch=15,
          batch_size=256):
    
    # load data
    X_pos_train,X_pos_val,X_neg_train,X_neg_val = data_loader(pos_train_path,pos_val_path,neg_path)
    
    # load pssm
    with open(pssm, 'rb') as f:
        motif_array = np.load(f)
    
    #number of total tf (half of pssm)
    ntf = motif_array.shape[-1]//2
    
    # motif_size
    motif_size = motif_array.shape[0]
    
    # padding size
    if ntf%4 ==0:
        npad = 0
    elif ntf%4 !=0:
        npad = 4 - ntf%4
    
    # sequence size
    sequence_size = X_pos_train.shape[1]
    
    # load model
    model= make_model_attn_cluster(num_tf=ntf,pad_size=npad,max_len=motif_size,pssm_path=pssm,lr=learning_rate,seq_size=sequence_size)

    history_list=[]
    eval_stats = []
    assert early_method in ['prc','auc'], "x is not equal to 5"
    for i in range(nepoch):
        X_train, y_train =trainSub(X_pos_train,X_neg_train,i)
        X_val, y_val =trainSub(X_pos_val,X_neg_val,i)
        history_current = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val))
        if early_stop:
            if early_method=='prc':
                eval_stats.append(history_current.history['val_prc'][0])
            elif early_method=='auc':
                eval_stats.append(history_current.history['val_auc'][0])
            if check_consecutive_difference(eval_stats,window_size= early_window,thres=early_thres,wait=early_wait ):
                break
        history_list.append(history_current)
    return model, history_list
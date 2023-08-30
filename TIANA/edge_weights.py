import numpy as np
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
import os
import glob
#from layers_helper import PositionalEncoding,AttentionBlock
sys.path.append('/gpfs/data01/glasslab/home/zhl022/daima/to_share/DeepLearningAttention/round3_code')
from model_layers import *
from pre_made_model import *
from ig_attn import *
from attn_weights import *
from sklearn.model_selection import train_test_split



class EdgeWeights:
    def __init__(self,
                 model_path,
                 pssm,
                 motif_cutoff_path,
                 our_npy_dir,
                 neg_path,
                 pos_train_path,
                 pos_val_path=None,
                 ncore=25,
                 score_only=False,
                 rank_or_ig='rank',
                 batch_size=1000,
                 neg_size=20000,
                 seq_size=200,
                 motif_threshold='e3'):
        self.model_path = model_path
        self.pssm=pssm
        self.motif_cutoff_path=motif_cutoff_path
        self.pos_train_path=pos_train_path
        self.pos_val_path=pos_val_path
        self.neg_path=neg_path
        self.ncore=ncore
        self.score_only =score_only
        self.rank_or_ig=rank_or_ig
        self.batch_size=batch_size
        self.seq_size=seq_size
        self.motif_threshold=motif_threshold
        self.our_npy_dir=our_npy_dir
        self.neg_size = neg_size
        #self.attn_method=None
        edge_np_rank_np=None
        # attn type score/ig & rank/ig
        if self.score_only:
            self.attn_method = self.rank_or_ig + "_" + "score"
        else: 
            self.attn_method = self.rank_or_ig + "_" + "ig"

        # load pssm, get first layer motif and rc (sum)
        with open(pssm, 'rb') as f:
            motif_array = np.load(f)
        self.first_filter_num= motif_array.shape[-1]
        # get motif size
        self.motif_size = motif_array.shape[0]
        # get padding size
        # first need to get ntf
        self.ntf = self.first_filter_num//2
        if self.ntf%4 ==0:
            self.npad = 0
        elif self.ntf%4 !=0:
            self.npad = 4 - self.ntf%4

        model_weights = load_model(self.model_path)
        self.model= make_model_attn_cluster(num_tf=self.ntf,
                                       pad_size=self.npad,
                                       max_len=self.motif_size,
                                       pssm_path=self.pssm,
                                       lr=5e-5,seq_size=self.seq_size)
        self.model.set_weights(model_weights.get_weights())
        self.model.trainable=False
        print('done loading model')

    def load_pos(self):
        with open(self.pos_train_path, 'rb') as f:
            X_pos_train = np.load(f)
        if self.pos_val_path is not None:
            with open(self.pos_val_path, 'rb') as f:
                X_pos_val= np.load(f)
            # assert val and train are same size
            assert X_pos_train.shape[1] == X_pos_val.shape[1], "error, train and val data must be same size"
            result_matrix = np.concatenate((X_pos_train, X_pos_val), axis=0)
        else:
            result_matrix=X_pos_train.copy()
        return result_matrix
    
    def load_neg(self):
        with open(self.neg_path, 'rb') as f:
            X_neg = np.load(f)
        return X_neg

    def batch_compute_pos(self):
        full_pos = self.load_pos()
        ncycle = 1 + full_pos.shape[0]//self.batch_size
        for i in range(ncycle):
            start = i*self.batch_size
            end = start +self.batch_size
            batch_pos=full_pos[start:end,...]
            if batch_pos.shape[0]>1:
                edge_list_rank=get_edge_list(self.model,
                                            batch_pos,
                                            method=self.rank_or_ig ,
                                            score_only=self.score_only,
                                            label=1,
                                            ncore=self.ncore,
                                            first_filter_num=self.first_filter_num,
                                            cpath=self.motif_cutoff_path,
                                            padding_size=self.npad,
                                            cutoff=self.motif_threshold,)
                edge_np_rank_np=np.array(edge_list_rank)
                if edge_np_rank_np.ndim ==1:
                    continue
                else:
                    # fix index on column 2 (3rd column)
                    edge_np_rank_np[:,2] += start


                    out_file_name = "pos_"+ self.attn_method + "_" + self.motif_threshold + "_batch_" + str(self.batch_size) + "_" + str(i)+".npy"
                    out_full_path = os.path.join(self.our_npy_dir,out_file_name)
                    with open(out_full_path, 'wb') as f:
                        np.save(f, edge_np_rank_np)
            else:
                continue
    
    def batch_compute_neg(self):
        full_neg = self.load_neg()
        # may only need first 20k of negative size 
        full_neg = full_neg[:self.neg_size,...]
        ncycle = 1 + full_neg.shape[0]//self.batch_size
        for i in range(ncycle):
            start = i*self.batch_size
            end = start +self.batch_size
            batch_neg=full_neg[start:end,...]
            
            if batch_neg.shape[0]>1:
                edge_list_rank=get_edge_list(self.model,
                                            batch_neg,
                                            method=self.rank_or_ig ,
                                            score_only=self.score_only,
                                            label=0,
                                            ncore=self.ncore,
                                            first_filter_num=self.first_filter_num,
                                            cpath=self.motif_cutoff_path,
                                            padding_size=self.npad,
                                            cutoff=self.motif_threshold,)
                edge_np_rank_np=np.array(edge_list_rank)
                if edge_np_rank_np.ndim ==1:
                    continue
                else:
                # fix index on column 2 (3rd column)
                    edge_np_rank_np[:,2] += start


                    out_file_name = "neg_" + self.attn_method + "_" +  self.motif_threshold + "_batch_" + str(self.batch_size) + "_" + str(i)+".npy"
                    out_full_path = os.path.join(self.our_npy_dir,out_file_name)
                    with open(out_full_path, 'wb') as f:
                        np.save(f, edge_np_rank_np)
            else:
                continue
    
    def merge_pos_npy(self):
        pattern = "pos_*.npy"
        npy_files  = os.path.join(self.our_npy_dir,pattern)
        concatenated_array = None
        for npy_file in sorted(glob.glob(npy_files)):
            with open(npy_file, 'rb') as f:
                loaded_array = np.load(f)
            if concatenated_array is None:
                concatenated_array = loaded_array
            else:
                concatenated_array = np.concatenate((concatenated_array, loaded_array), axis=0)
        output_filename = "full_pos_edge_" + self.attn_method +".npy"
        output_path = os.path.join(self.our_npy_dir, output_filename) 
        with open(output_path, 'wb') as f:
            np.save(f, concatenated_array)             

    def merge_neg_npy(self):
        pattern = "neg_*.npy"
        npy_files  = os.path.join(self.our_npy_dir,pattern)
        concatenated_array = None
        for npy_file in sorted(glob.glob(npy_files)):
            with open(npy_file, 'rb') as f:
                loaded_array = np.load(f)
            if concatenated_array is None:
                concatenated_array = loaded_array
            else:
                concatenated_array = np.concatenate((concatenated_array, loaded_array), axis=0)
        output_filename = "full_neg_edge_" + self.attn_method +".npy"
        output_path = os.path.join(self.our_npy_dir, output_filename) 
        with open(output_path, 'wb') as f:
            np.save(f, concatenated_array) 
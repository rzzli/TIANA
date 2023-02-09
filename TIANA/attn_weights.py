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
from pre_made_model import *
from ig_attn_class import *
from sklearn.model_selection import train_test_split
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from itertools import combinations
from statannot import add_stat_annotation
from scipy.stats import rankdata
#import multiprocessing
import concurrent

#This module computes attention attributes or scores from fully trained model, the output contains:
#   combination_index, rank/ig/score, peak_idx, motif1 ,motif2 , loc1,loc2
#


class ComputePairRank:
    def __init__(self,full_model,data_X_to_predict, label, score_only=False, steps=100, first_filter_num= 560, dropout_rate=0.4,attn_layer=8,
                cutoff='e4', padding_size=15,
                cpath='/home/zhl022/daima/projects/dl_methods_improve/motif_pssm_Aug27/motif_npy/motif265_name_cutoff_Sep5.npy'):
        self.full_model=full_model
        self.data_X_to_predict=data_X_to_predict
        self.first_filter_num=first_filter_num
        self.dropout_rate=dropout_rate
        self.attn_layer=attn_layer
        self.label=label
        self.score_only=score_only
        self.steps=steps
        self.cutoff=cutoff
        self.cpath=cpath
        self.padding_size=padding_size

        self.ig_attn=[]
        self.ig_attn_max=None
        self.model_att=None
        self.att_out=None
        self.scores=None
        self.myvalue=None
        self.model_attn_ig=None
        self.pass_cut_mtx=None
        self.pos_loc_mtx=None
        self.pre_att_out=None
        self.ig_i_obj=None

    def step1_compute_attn_score(self):
        attn_layer_before = self.attn_layer-1
        model_pre_att_output=tf.keras.models.Model(inputs=self.full_model.inputs, outputs=self.full_model.layers[attn_layer_before].output)
        
        self.pre_att_out =model_pre_att_output.predict(self.data_X_to_predict) 
        
        self.model_att= tf.keras.models.Model(inputs=self.full_model.inputs, outputs=self.full_model.layers[self.attn_layer].output)
        
        self.att_out,self.scores,self.myvalue =self.model_att.predict(self.data_X_to_predict) 
        #return self.att_out,self.scores,self.myvalue,self.model_att

    def step2_make_attn_after_model(self):
        #att_out,scores,myvalue,model_att=self.step1_compute_attn_score()
        inputs = layers.Input(shape=self.scores.shape[1:])
        #pre_layer= AttentionGrad_preLayer(value_shape=self.myvalue.shape[1:]) Dec 4 edits
        #pre_layer= AttentionGrad_preLayer(myvalue=self.myvalue) #Dec 4 edits
        pre_layer= AttentionGrad_preLayer(value_shape=self.myvalue.shape[1:]) # Dec 5 edits
        x=pre_layer(inputs)
        attg=AttentionGrad(embed_dim=self.first_filter_num/2,num_heads=4, ff_dim=self.first_filter_num,
                    #pre_input_dim=self.pre_att_out.shape[1:], # Dec 4
                    #pre_input=self.pre_att_out,# Dec 4 edits
                    pre_input_dim=self.pre_att_out.shape[1:], #Dec 5
                    mykernel=self.model_att.layers[-1].att.projection_kernel,
                    mybias=self.model_att.layers[-1].att.projection_bias )
        x  = attg(x)
        x = layers.Flatten()(x)
        x=layers.Dropout(self.dropout_rate)(x)
        x=layers.Dense(256,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
            kernel_regularizer=keras.regularizers.l2(5e-7),
            bias_initializer=keras.initializers.Constant(value=0),
            activity_regularizer=keras.regularizers.l1(1e-8))(x)
        x=layers.Activation('relu')(x)
        x=layers.Dropout(self.dropout_rate)(x)
        x=layers.Dense(64,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
            kernel_regularizer=keras.regularizers.l2(5e-7),
            bias_initializer=keras.initializers.Constant(value=0),
            activity_regularizer=keras.regularizers.l1(1e-8))(x)
        x=layers.Activation('relu')(x)
        """
        x=layers.Dense(1,    
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=1e-2),
            kernel_regularizer=keras.regularizers.l2(5e-7),
            bias_initializer=keras.initializers.Constant(value=0),
            activity_regularizer=keras.regularizers.l1(1e-8))(x)
        #outputs = layers.Activation('sigmoid')(x) Dec 4  edits
        x=layers.Activation('relu')(x)""" # Dec4 edits
        outputs=layers.Dense(1, activation='sigmoid',bias_initializer=None)(x)
        self.model_attn_ig = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer_learning=tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.model_attn_ig.compile(optimizer=optimizer_learning, loss="binary_crossentropy", metrics=["accuracy"])
        self.model_attn_ig.trainable=False


        #return self.model_attn_ig
        # Dec 4 assign weights
        for i in range(len(self.full_model.layers[self.attn_layer+1:])):
            self.model_attn_ig.layers[3:][i].set_weights(self.full_model.layers[self.attn_layer+1:][i].weights)
        self.model_attn_ig.layers[2].ffn.set_weights(self.full_model.layers[self.attn_layer].ffn.weights)
        self.model_attn_ig.layers[2].layernorm1.set_weights(self.full_model.layers[self.attn_layer].layernorm1.weights)
        self.model_attn_ig.layers[2].layernorm2.set_weights(self.full_model.layers[self.attn_layer].layernorm2.weights)
        self.model_attn_ig.layers[2].batchnorm1.set_weights(self.full_model.layers[self.attn_layer].batchnorm1.weights)
        self.model_attn_ig.layers[2].batchnorm2.set_weights(self.full_model.layers[self.attn_layer].batchnorm2.weights)
    #def step3_compute_ig_attn(self):
    def step3a_compute_ig_attn(self):

        baseline= np.zeros(self.scores.shape[1:])
        for i in range(self.scores.shape[0]):
            """
            current_ig = integrated_gradients_attn_layer(self.scores[i,...],
                                                        mymodel=self.model_attn_ig,
                                                        baseline_freq=baseline,
                                                        myvalue_full=self.myvalue,
                                                        my_pre_input_full=self.pre_att_out,
                                                        current_i=i,
                                                        label=self.label,
                                                        m_steps=self.steps
                                                        )
            """
            self.ig_i_obj=None
            self.ig_i_obj=IgAttn(self.scores[i,...],
                            mymodel=self.model_attn_ig,
                            baseline_freq=baseline,
                            myvalue_full=self.myvalue,
                            my_pre_input_full=self.pre_att_out,
                            current_i=i,
                            label=self.label,
                            m_steps=self.steps
                            )
            current_ig=self.ig_i_obj.integrated_gradients_attn_layer()
            self.ig_attn.append(current_ig) 
        self.ig_attn_max = np.array([x.numpy().max(axis=0) for x in self.ig_attn])  # max over 4 attn heads 

        #return self.ig_attn,self.ig_attn_max
    def step3b_compute_score_attn(self):

        self.ig_attn_max = np.array([x.max(axis=0) for x in self.scores])  # max over 4 attn heads 

    def step4_motif_cut_off(self):
        def get_cutoff():

            with open(self.cpath,'rb') as f:
                motif_name = np.load(f)
                motif_cutoffe4 = np.load(f)
                motif_cutoffe3 = np.load(f)
            motif_cutoffe4_pad=np.append(motif_cutoffe4, [1]*self.padding_size)
            motif_cutoffe3_pad=np.append(motif_cutoffe3, [1]*self.padding_size)
            return motif_name,motif_cutoffe4_pad,motif_cutoffe3_pad

        def compute_pass_cutoff(l2maxp,cutoff=self.cutoff):
            """
            input: maxpool output of layer[2], should be (nsample,279,280)
            input (optional): which cutoff to use, default e4

            output: maxpooled array, (nsample,70,280)
            """
            motif_name,motif_cutoffe4_pad,motif_cutoffe3_pad = get_cutoff()
            if self.cutoff=='e3':
                motif_cutoff_pad=motif_cutoffe3_pad
            else:
                motif_cutoff_pad=motif_cutoffe4_pad

            #pass_cutoff_bool=((l2maxp-motif_cutoff_pad)>= 0).astype(int)  # {0,1} mtx of (nsample,279,280)
            pass_cutoff_bool=((np.round(l2maxp,3)-np.round(motif_cutoff_pad,3)) >  -0.0001).astype(int)  # Dec 2 edit to avoid float32 - float64 preision error
            pass_cutoff_maxp=layers.MaxPooling1D(pool_size=4, strides=4, padding='same')(pass_cutoff_bool) # (nsample,70,280)
            pass_cutoff_maxp_np=pass_cutoff_maxp.numpy()
            return pass_cutoff_maxp_np
        
        model_maxpoll_output=tf.keras.models.Model(inputs=self.full_model.inputs, outputs=self.full_model.layers[2].output)
        maxp_out =model_maxpoll_output.predict(self.data_X_to_predict) 
        self.pass_cut_mtx=compute_pass_cutoff(maxp_out)
        self.pos_loc_mtx=(self.pass_cut_mtx.sum(axis=2)> 0).astype(int) #{0,1} nsample x 70
        #return self.pass_cut_mtx, self.pos_loc_mtx
    def step5_compute_ig_and_cutoff(self):
        if self.score_only==False:
            self.step1_compute_attn_score()
            self.step2_make_attn_after_model()
            self.step3a_compute_ig_attn()
            self.step4_motif_cut_off()
            return self.ig_attn_max,self.pass_cut_mtx
        elif self.score_only==True:
            self.step1_compute_attn_score()
            self.step2_make_attn_after_model()
            self.step3b_compute_score_attn()
            self.step4_motif_cut_off()
            return self.ig_attn_max,self.pass_cut_mtx

class CompEdges:
    def __init__(self,pass_cut_mtx,ig_mtx,method='rank',ncore=1):
        # need input rank matrix
        #pass_cut_mtx.shape 
        # rank or ig
        #tfnum=280
        #tfcom=list(combinations(range(tfnum),2))
        self.pass_cut_mtx=pass_cut_mtx
        self.ig_mtx=ig_mtx #(nsample,70,70)
        self.tfnum=pass_cut_mtx.shape[-1]
        self.tfcom=[tuple(sorted(p)) for p in list(combinations(range(self.tfnum),2))]
        self.method=method
        self.ncore=ncore
        self.nsample_sym=None
        self.value_list=[[] for x in range(len(self.tfcom))]
        self.score_dict={}
        
        ### need operation
        ### 1. decide which mtx to move forward # rank or ig
        ### 2. max over diag
        ### 3. for all nsamples
    ######
    #sample prep
    ######
    def max_over_diag(self,mtx1):
        if self.method=='rank':
            ranked_mtx1=rankdata(mtx1).reshape(mtx1.shape)
            ranked_mtx1_sym = np.maximum(ranked_mtx1, ranked_mtx1.transpose() )
            return ranked_mtx1_sym.astype(int)
        elif self.method=='ig':
            mtx1_sym = np.maximum(mtx1, mtx1.transpose() )
            return mtx1_sym
    def get_nsample_diag(self):
        self.nsample_sym=np.array([self.max_over_diag(self.ig_mtx[i,...]) for i in range(self.ig_mtx.shape[0])]) ## nsample,70,70
    
    ######
    #compute 
    ######
    '''
    def add_element(self,mydict, key, value):
        if key not in mydict:
            mydict[key] = []
        mydict[key].append(value)
        
    def merge_dicts(self,main_dict, new_dict ):
        new_dict = copy.deepcopy(main_dict)
        for key, value in new_dict.items():
            new_dict.setdefault(key, []).extend(value)
        self.score_dict=new_dict
    '''    
    def loc_pair_add_attn(self,pair_info_row):
        #current_idx=pair_info_row[3]
        #pair0=pair_info_row[0]
        #pair1=pair_info_row[1]
        #pair_score=pair_info_row[2]
        pair0,pair1,pair_score,current_idx =pair_info_row
        if self.method=='ig':
            pair0=int(pair0)
            pair1=int(pair1)
            current_idx=int(current_idx)
        motif_pair_list=list(itertools.product(np.where(self.pass_cut_mtx[current_idx,pair0,:])[0],np.where(self.pass_cut_mtx[current_idx,pair1,:])[0]))
        #motif_pair_list=[tuple(sorted(p)) for p in motif_pair_list if p[0]!=p[1]]
        #return [[self.tfcom.index(p),pair_score,current_idx] for p in motif_pair_list]
        motif_pair_list=[[tuple(sorted(p)),p[0],p[1]] for p in motif_pair_list if p[0]!=p[1]]
        #return [[self.tfcom.index(p[0]),pair_score,current_idx,p[1],p[2]] for p in motif_pair_list]  
        return [[self.tfcom.index(p[0]),pair_score,current_idx,p[1],p[2],pair0,pair1] for p in motif_pair_list] # edit Sep 20 to long lont
            # combination_index, rank/ig, peak_idx, motif1,motif2, loc1,loc2

    def flatten(self,l):
        return [item for sublist in l for item in sublist]

    def loc_pairs_cur_sample(self,idx):
        current_pass_cut_mtx=self.pass_cut_mtx[idx,...]
        pair_pos=np.where(current_pass_cut_mtx.sum(axis=-1)>0)[0]  # identify loc_index with none-empty motif
        loc_pairs=np.array(list(combinations(pair_pos,2))) # (xcomb,2) shaped np array with combination of none empty index # out of 70
        npairs=loc_pairs.shape[0]
        attn_scores=np.array([self.nsample_sym[idx,loc_pair[0],loc_pair[1]] for loc_pair in loc_pairs])[:,np.newaxis]
        try:
            pair_info_mtx=np.hstack((loc_pairs,  attn_scores,np.repeat(idx,npairs)[:,np.newaxis]))  #(npair,4) array mtx
            #return [self.loc_pair_add_attn(pair_info_mtx[i,:]) for i in range(pair_info_mtx.shape[0])]
            return pair_info_mtx
        except:
            return
    
    def all_sample_arr(self):
        self.get_nsample_diag()
        ## returns [[a/70, b/70, ig/rank,  idx/peak_num],...]
        #return self.flatten([ self.loc_pairs_cur_sample(idx) for idx in range(self.pass_cut_mtx.shape[0])])
        all_sample_pair_list=[ self.loc_pairs_cur_sample(idx) for idx in range(self.pass_cut_mtx.shape[0])]
        return self.flatten(list(filter(lambda item: item is not None, all_sample_pair_list)))
    def get_res(self):
        all_pair_info_mtx=self.all_sample_arr()
        #with multiprocessing.Pool(self.ncore) as pool:
            #self.mtxA=np.array(list(pool.map(self.loc_pair_add_attn, all_pair_info_mtx))) #list(map(lambda i:ce5.loc_pair_add_attn(out5[i]) , range(1000)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.ncore ) as executor:
            self.mtxA = np.array(list(executor.map(self.loc_pair_add_attn, all_pair_info_mtx,chunksize=20000)))
        return self.flatten(self.mtxA)

def get_edge_list(full_model,
                    data_X_to_predict, 
                    label,
                    score_only=False,
                    ncore=1,
                    steps=50, 
                    first_filter_num= 560, 
                    dropout_rate=0.4,
                    method='rank',
                    attn_layer=8,
                    cpath='/home/zhl022/daima/projects/dl_methods_improve/motif_pssm_Aug27/motif_npy/motif265_name_cutoff_Sep5.npy',
                    cutoff='e4',
                    padding_size=15):
    cpr=ComputePairRank(full_model,
                        data_X_to_predict ,
                        label=label,
                        score_only=score_only, # if score_only is True, may wannt use method = 'ig'
                        steps=steps, 
                        first_filter_num=first_filter_num,
                        dropout_rate=dropout_rate,
                        attn_layer=attn_layer,
                        cutoff=cutoff,
                        cpath=cpath,
                        padding_size=padding_size)
    ig_attn_max, pass_cut_mtx = cpr.step5_compute_ig_and_cutoff()
    ce=CompEdges(pass_cut_mtx=pass_cut_mtx,
                    ig_mtx=ig_attn_max,
                    method=method,
                    ncore=ncore)
    out_mtx=ce.get_res()
    return out_mtx


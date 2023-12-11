import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from random import choices, sample
from itertools import combinations
from scipy import stats
import tensorflow as tf
from tensorflow import keras
import time
import random
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata, mannwhitneyu
from attn_weights import *

tf.random.set_seed(1)


#effect size
def effectSize(arr1,arr2):
    #https://psycnet.apa.org/fulltext/2011-16756-001.pdf?auth_token=db2b224ec041f68681ca3072df4ab2963e80a9e8&returnUrl=https%3A%2F%2Fpsycnet.apa.org%2Frecord%2F2011-16756-001 
    #page 12
    #https://datatab.net/tutorial/mann-whitney-u-test
    n1=len(arr1)
    n2=len(arr2)
    T1=sum(arr1) 
    T2=sum(arr2) 
    U1=(n1*n2)+(n1*(n1+1)/2)-T1
    U2=(n1*n2)+(n2*(n2+1)/2)-T2
    U=np.minimum(U1,U2)
    expU = n1 * n2 /2
    steU = np.sqrt( n1* n2 *(n1+n2+1)/12)
    z=(U-expU)/steU
    r= np.abs(z)/np.sqrt(n1+n2)
    return r

class EdgeMotif:
    def __init__(self,                 
                 motif_cutoff_path,
                 pos_edge_path,
                 neg_edge_path,
                 pssm,
                 tf_family_map,
                 seq_length=200,
                 pos_npeak_min=0.05,
                 attn_min=0.95,
                 min_effect=10,
                 pcutoff=1e-4
                 
                 
                 ):
        self.pos_edge_path=pos_edge_path
        self.neg_edge_path=neg_edge_path
        self.motif_cutoff_path=motif_cutoff_path
        
        # load edge matrix
        with open(self.pos_edge_path,'rb') as f:
            self.pos_edge_mtx = np.load(f)
        with open(self.neg_edge_path,'rb') as f:
            self.neg_edge_mtx = np.load(f)
            
        # load motif name and cutoff
        with open( self.motif_cutoff_path,'rb') as f:
            self.motif_name = np.load(f)
            _ = np.load(f)
            _ = np.load(f)
            
        # compute motif length (22)
        with open(pssm, 'rb') as f:
            motif_array = np.load(f)
        self.tfn = motif_array.shape[-1]//2 #224
        self.tf_size = motif_array.shape[0] 
        assert isinstance(self.tfn, int), "tfn is not an integer"

        
        # compute location 45 for 200np and 70 for 300bp
        self.nloc = math.ceil((seq_length+1-self.tf_size)/4) #45
        
        #load self.tfcom
        self.tfcom=list(combinations(range(self.tfn),2))
        
        #load tf family info
        tf_map_df=pd.read_csv(tf_family_map)
        self.tf_map_dict={}
        for index, row in tf_map_df.iterrows():
            self.tf_map_dict[row[0]]=str(row[1]).replace("\xa0","")

        self.pos_npeak_min=pos_npeak_min #0.05
        self.attn_min=attn_min #0.95
        self.min_effect=min_effect #10
        self.pcutoff=pcutoff

        self.res_df=None
        self.res_df_sub=None
        self.res_df_sub_short_redo_3col=None
        self.res_df_sub_short=None
        self.res_df_sub_short_redo_all_col =None

        # Dec 2023 add max to retrive motif loc
        self.pos_edge_mtx_df_max_np=None
        self.neg_edge_mtx_df_max_np=None
        
    def get_tfs_spacing(self,row):
        # compute spacing for self.res_df_sub_short_redo_all_col
        tf1_name = row['tf1_name']
        tf2_name = row['tf2_name']
        tf1_idx = np.where(self.motif_name==tf1_name)[0][0]
        tf2_idx = np.where(self.motif_name==tf2_name)[0][0]
        # short list of [tf1_idx,tf2_idx]
        tf_idx_match = [ tf1_idx,tf2_idx ]
        # subset mtx
        outmtx= self.pos_edge_mtx_df_max_np[np.isin(self.pos_edge_mtx_df_max_np[:, 3], tf_idx_match) & (np.isin(self.pos_edge_mtx_df_max_np[:, 4], tf_idx_match)),:] 
        # subset tf1,tf2 left and right
        tf1_left = outmtx[ outmtx[:,3]==tf1_idx,:]  # equals tf2_right
        tf1_right = outmtx[ outmtx[:,4]==tf1_idx,:]  #equals tf2_left
        # loc
        tf1_loc = np.concatenate([tf1_left[:,5],tf1_right[:,6]])
        tf2_loc = np.concatenate([tf1_left[:,6],tf1_right[:,5]])
        tf1_loc_expand = (tf1_loc*4)+21
        tf2_loc_expand = (tf2_loc*4)+21
        spacing= np.abs(tf1_loc_expand - tf2_loc_expand)
        mean_bp = int(np.mean(spacing) )
        std_bp = int(np.std(spacing) )
        return (mean_bp,std_bp)   
    
    def mergeRows(self):
        self.neg_edge_mtx=self.neg_edge_mtx[self.neg_edge_mtx[:,-1]!=self.neg_edge_mtx[:,-2],:]
        self.pos_edge_mtx=self.pos_edge_mtx[self.pos_edge_mtx[:,-1]!=self.pos_edge_mtx[:,-2],:]

        neg_edge_mtx_df_max=pd.DataFrame(self.neg_edge_mtx,columns =['combination_index', 'rank/ig', 'peak_idx', 'motif1','motif2', 'loc1','loc2'])
        neg_edge_mtx_df_max=neg_edge_mtx_df_max.loc[neg_edge_mtx_df_max.groupby(['combination_index',"peak_idx"])["rank/ig"].idxmax(),:]

        pos_edge_mtx_df_max=pd.DataFrame(self.pos_edge_mtx,columns =['combination_index', 'rank/ig', 'peak_idx', 'motif1','motif2', 'loc1','loc2'])
        #xdf.loc[xdf.groupby(['combination_index',"peak_idx"])["rank/ig"].idxmax(),:]
        pos_edge_mtx_df_max=pos_edge_mtx_df_max.loc[pos_edge_mtx_df_max.groupby(['combination_index',"peak_idx"])["rank/ig"].idxmax(),:]

        self.neg_edge_mtx_df_max_np=neg_edge_mtx_df_max.to_numpy()
        self.pos_edge_mtx_df_max_np=pos_edge_mtx_df_max.to_numpy()

        
        def process_data(i):
            pos_has = self.pos_edge_mtx_df_max_np[self.pos_edge_mtx_df_max_np[:, 0] == i]
            neg_has = self.neg_edge_mtx_df_max_np[self.neg_edge_mtx_df_max_np[:, 0] == i]

            if pos_has.shape[0] * neg_has.shape[0] == 0:
                return None

            _, p = mannwhitneyu(pos_has[:, 1], neg_has[:, 1])
            effect_size = effectSize(pos_has[:, 1], neg_has[:, 1])
            mean_pos = np.mean(pos_has[:, 1])
            mean_neg = np.mean(neg_has[:, 1])

            return [i, mean_pos, mean_neg, pos_has.shape[0], neg_has.shape[0], effect_size, p]
        
        indices = range(len(self.tfcom))
        res_list = list(map(process_data, indices))
        res_list = [entry for entry in res_list if entry is not None]

        if self.tfn == 224:
            # tal1 if in 224 motifs
            pair195=[]
            for i,pair in enumerate(self.tfcom):
                if 195 in pair:
                    pair195.append(i)
            res_list_np=np.array(res_list)
            res_list_np=res_list_np[~np.isin(res_list_np[:,0],pair195),:]
        else:
            res_list_np=np.array(res_list)

        self.res_df=pd.DataFrame(res_list_np,columns =['combination_id',
                                                'mean_pos',
                                                'mean_neg',
                                                'count_pos',
                                                'count_neg',
                                                'effect_size',
                                                'p'])

        self.res_df['tf1_idx']=self.res_df.apply( lambda x: self.tfcom[x["combination_id"].astype(int)][0] , axis=1)
        self.res_df['tf2_idx']=self.res_df.apply( lambda x: self.tfcom[x["combination_id"].astype(int)][1] , axis=1)
        self.res_df['tf1_name']=self.res_df.apply( lambda x: self.motif_name[int(x['tf1_idx'] )], axis=1)
        self.res_df['tf2_name']=self.res_df.apply( lambda x: self.motif_name[int(x['tf2_idx'] )], axis=1)
        self.res_df['pos_peakN']=len(set(self.pos_edge_mtx[:,2]))
        self.res_df['neg_peakN']=len(set(self.neg_edge_mtx[:,2]))
        self.res_df['pos_perc']=100*self.res_df['count_pos']/self.res_df['pos_peakN']
        self.res_df['neg_perc']=100*self.res_df['count_neg']/self.res_df['neg_peakN']

        self.res_df["tf1_name_short"]=self.res_df.apply(lambda x:  str(x["tf1_name"]).split('|')[0]  ,axis=1  )
        self.res_df["tf2_name_short"]=self.res_df.apply(lambda x:  str(x["tf2_name"]).split('|')[0]  ,axis=1  )
        self.res_df["mean_pos_percentile"]=self.res_df["mean_pos"]/(self.nloc**2)
        self.res_df["mean_neg_percentile"]=self.res_df["mean_neg"]/(self.nloc**2)
        self.res_df["adjp"]=-np.log10(self.res_df["p"]+1e-1000)
        self.res_df["adjp"] = self.res_df["adjp"] 
        npeak=self.res_df.pos_peakN.max()

        count_cutoff=npeak*self.pos_npeak_min
        rank_cutoff=(self.nloc**2) * self.attn_min
        effectsize_cutoff= self.min_effect

        self.res_df_sub=self.res_df[(self.res_df['count_pos']>count_cutoff)&
                        (self.res_df['mean_pos']>rank_cutoff)&
                        (self.res_df['p']*len(self.tfcom)<self.pcutoff) & 
                        ((self.res_df['effect_size']>effectsize_cutoff))].sort_values(['p'],ascending=True)

        self.res_df_sub["tf1_family"]=self.res_df_sub.apply(lambda x: self.tf_map_dict[str(x["tf1_name"])]  ,axis=1  )
        self.res_df_sub["tf2_family"]=self.res_df_sub.apply(lambda x: self.tf_map_dict[str(x["tf2_name"])]  ,axis=1  )
        self.res_df_sub_short = self.res_df_sub.loc[:,['combination_id',
                                            'mean_pos_percentile',
                                            'mean_neg_percentile', 
                                            'count_pos', 
                                            'count_neg',
                                            'p', 
                                            'pos_perc', 
                                            'neg_perc', 
                                            'tf1_family', 
                                            'tf2_family', 
                                            'tf1_name_short', 
                                            'tf2_name_short',
                                            'tf1_name', 
                                            'tf2_name',
                                            'adjp',
                                            'effect_size']]
        self.res_df_sub_short['tf1tf2_family']=self.res_df_sub_short.apply(lambda x: tuple(sorted((x["tf1_family"],x["tf2_family"]))), axis =1)
        self.res_df_sub_short_redo_3col=self.res_df_sub_short.groupby(['tf1tf2_family']).agg({"pos_perc":"max"}).reset_index()
        good_idx_list=[]
        for index, row in self.res_df_sub_short_redo_3col.iterrows():
            good_idx=self.res_df_sub_short[(self.res_df_sub_short['tf1tf2_family']==row['tf1tf2_family']) &
                                      (self.res_df_sub_short['pos_perc']==row['pos_perc'])].index[0]
            good_idx_list.append(good_idx)
        self.res_df_sub_short_redo_all_col=self.res_df_sub_short.loc[good_idx_list,:]
        self.res_df_sub_short_redo_all_col=self.res_df_sub_short_redo_all_col.sort_values(['adjp','effect_size','pos_perc'],ascending=False)
        self.res_df_sub_short_redo_all_col.reset_index(inplace=True)
        self.res_df_sub_short_redo_all_col['spacing (mean,std)']=self.res_df_sub_short_redo_all_col.apply(self.get_tfs_spacing,axis=1)
        return self.res_df_sub_short_redo_all_col
        
    def generate_html(self,result_df_path=None, html_dir='./'):
        if result_df_path is not None:
            df =  pd.read_csv(result_df_path,index_col=0)
        else:
            df = self.res_df_sub_short_redo_all_col
        assert df is not None, "result dataframe cannot be empty"
        if df.shape[0]>30:
            df=df.iloc[:30,]
        
        
                 # universal designs
        header = '<head>\n<title>TNASA results</title>\n</head>'
        basic_setup = '<body bgcolor="white" text="black">\n<table border="2" align="center" width="50%" height="20%" bordercolor="grey" cellspacing="5" cellpadding="20">'
        caption = '<caption><font size="6", color="green"><b>Significant functional TF motif pairs </b></font><BR><font size="4", color="black">' 
        label_color = '#1387FF'
        table_label = '<tr>\n<th width="10%" height="4%"><font color="'

        table_label += label_color
        table_label += '">Rank</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">motif1|family</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">motif1_logo</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">motif2|family</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">motif2_logo</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">attention in target sequences</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">attention in background sequences</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">motif pair percent in target sequences</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">motif pair percent in background sequences</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">spacing (mean,std) bp</font></th>\n<th><font color="'
        table_label += label_color
        table_label +='">p value(-log10)</font></th>\n</tr>'
        ending = '</table>\n</body>\n</html>\n'
        
        ## html_path 
        html_path = os.path.join(html_dir,"TIANA_result.html")
        with open(html_path, 'w') as hf:
            hf.write('<html>\n')
            hf.write('\n'.join([header, basic_setup, caption, table_label]))
            idx=1
            for index, row in df.iterrows():
                #motif1_pre = 'img_tag = '<img src="logos/'+str(k+1)+'.png" width="350"/>'" + str(row['tf1_name'])+".png"
                #img1_tag = '<img src="'+ motif1_pre+'" width="200"/>'
                #motif2_pre = "/home/zhl022/daima/projects/Jan_tnasa/html/logo_html/" + str(row['tf2_name'])+".png"
                #img2_tag = '<img src="'+ motif2_pre+'" width="200"/>'
                #img_tag1 = '<img src="/home/zhl022/daima/projects/Jan_tnasa/html/logo_html/'+str(row['tf1_name'])+'.png"/>'
                #img_tag2 = '<img src="/home/zhl022/daima/projects/Jan_tnasa/html/logo_html/'+str(row['tf1_name'])+'.png"/>'
                img_tag1 = '<img src="logo_html/'+str(row['tf1_name'])+'.png" width="200" >'
                img_tag2 = '<img src="logo_html/'+str(row['tf2_name'])+'.png" width="200" >'

                rank=str(idx)
                idx+=1

                motif1_name = str(row["tf1_name_short"]) + "|" + str(row["tf1_family"]) 
                motif2_name = str(row["tf2_name_short"]) + "|" + str(row["tf2_family"]) 

                pos_rank_perc=str(np.round(row["mean_pos_percentile"],3))
                neg_rank_perc=str(np.round(row["mean_neg_percentile"],3))

                pos_perc_motif=str(np.round(row["pos_perc"],3))
                neg_perc_motif=str(np.round(row["neg_perc"],3))
                
                space_tf1tf2=str(row["spacing (mean,std)"])

                p_str= str(np.round(row["adjp"],3))


                #one_row = '<tr>\n<th>'+str(k+1)+'</th>\n
                one_row=''

                one_row += '<th>'
                one_row += rank
                one_row += '</th>\n'

                one_row += '<th>'
                one_row += motif1_name
                one_row += '</th>\n'

                one_row += '<th>'
                one_row += img_tag1
                one_row += '</th>\n'

                one_row += '<th>'
                one_row += motif2_name
                one_row += '</th>\n'

                one_row += '<th>'
                one_row += img_tag2
                one_row += '</th>\n'

                one_row += '<th>'
                one_row += pos_rank_perc
                one_row += '</th>\n'

                one_row += '<th>'
                one_row += neg_rank_perc
                one_row += '</th>\n'

                one_row += '<th>'
                one_row += pos_perc_motif
                one_row += '</th>\n'

                one_row += '<th>'
                one_row += neg_perc_motif
                one_row += '</th>\n'
                
                one_row += '<th>'
                one_row += space_tf1tf2
                one_row += '</th>\n'
                
                one_row += '<th>'
                one_row += p_str
                one_row += '</th>\n'

                one_row += '</tr>\n'
                #one_row2 = '<tr>\n<th>'+rank+'</th>\n<th>'+motif1_name+'</th>\n<th>'+img_tag1+'</th>\n<th>\n'+motif2_name+'\n'+img_tag2+'\n</th>\n<th>'+num_mut_block+'</th>\n<th>'+pos_mut_block+'</th>\n<th>'+neg_mut_block+'</th>\n<th>'+median_diff+'</th>\n<th>'+mean_diff+'</th>\n<th>'+img_distr+'</th>\n</tr>\n
                hf.write(one_row)
            hf.write(ending)

import logomaker as lm
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def make_logo(pssm,motif_cutoff_path,pre_path):
    with open(pssm, 'rb') as f:
        pssm = np.load(f)
    with open(motif_cutoff_path,'rb') as f:
        motif_name = np.load(f)
        motif_cutoffe4 = np.load(f)
        motif_cutoffe3 = np.load(f)
    #number of tf
    ntf = len(motif_name)
    #mkdir if not exist
    os.makedirs(pre_path, exist_ok=True) 
    def plotLogo(i,pre_path=pre_path):
        current_motif_pssm=pssm[...,i*2]
        current_motif_pssm_short=current_motif_pssm[((current_motif_pssm==0).sum(axis=1)!=4),]
        current_motif_pssm_dict={'A':current_motif_pssm_short[:,0].tolist() ,'C':current_motif_pssm_short[:,1].tolist(),'G':current_motif_pssm_short[:,2].tolist(),'T':current_motif_pssm_short[:,3].tolist()}
        display_df = pd.DataFrame(current_motif_pssm_dict)
        display_df = display_df[['A', 'C', 'G', 'T']]
        display_df.index.name = 'pos'
        display_df = display_df*(display_df>0)

        sns.set(style='white', font_scale=1.5)
        logo = lm.Logo(df=display_df,
                       font_name='DejaVu Sans',
                       fade_below=0.8,
                       shade_below=0.1, figsize=(6,2))
        sns.despine(top=True, bottom=True)
        file_path= pre_path+'/' + motif_name[i] +".png"
        plt.savefig(file_path,bbox_inches = 'tight')
    for i in range(ntf):
        plotLogo(i)
    print("done making logos at",pre_path )
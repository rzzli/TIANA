import numpy as np
import pandas as pd

def make_rc(pssm_path,name_path,df_path):
    """
    This function converts input motifs for reverse complement.
    Input: pssm, name and df path
    Compute and write rc files to the same folder as input
    Output: pssm, name and df path that are reverse complement included
    """
    with open(name_path, 'rb') as f:
        motif_name = np.load(f)
        motif_cutoff_e4 = np.load(f)
        motif_cutoff_e3 = np.load(f)
    with open(pssm_path, 'rb') as f:
        padded_motif_np_swap= np.load(f)
        
    # convert pssm (double)
    empty_array_pssm = np.empty((22, 4, 0))
    nmotif = padded_motif_np_swap.shape[2]
    for i in range(nmotif):
        current_motif = padded_motif_np_swap[:,:,i] 
        for _ in range(2):
            empty_array_pssm = np.append(empty_array_pssm, current_motif[:, :, np.newaxis], axis=2)
    # conver name 
    motif_name_wrc = np.array([])
    motif_cutoff_e4_wrc = np.array([])
    motif_cutoff_e3_wrc = np.array([])
    for i in range(motif_name.shape[0]):
        name_f = motif_name[i]
        name_r = motif_name[i]+ "_reverse"
        motif_name_wrc = np.append(motif_name_wrc, name_f)
        motif_name_wrc = np.append(motif_name_wrc, name_r)
        for _ in range(2):
            motif_cutoff_e4_wrc=np.append(motif_cutoff_e4_wrc, motif_cutoff_e4[i])
            motif_cutoff_e3_wrc=np.append(motif_cutoff_e3_wrc, motif_cutoff_e3[i])
    # convert df
    df = pd.read_csv(df_path,header=None)
    duplicated_df = df.loc[df.index.repeat(2)].reset_index(drop=True)
    for i in range(1, len(duplicated_df), 2):
        duplicated_df.at[i, 0] += '_reverse'
        
    # replace names 
    pssm_path_reverse = pssm_path.replace(".npy","_reverse.npy")
    name_path_reverse = name_path.replace(".npy","_reverse.npy")
    df_path_reverse = df_path.replace(".csv","_reverse.csv")
    
    # name out
    with open(name_path_reverse, 'wb') as f:
        np.save(f,np.array(motif_name_wrc))
        np.save(f,np.array(motif_cutoff_e4_wrc))
        np.save(f,np.array(motif_cutoff_e3_wrc))
    # pssm out
    with open(pssm_path_reverse, 'wb') as f:
        np.save(f,empty_array_pssm)
        
    # df out
    duplicated_df.to_csv(df_path_reverse, index = False, header=False)
    return pssm_path_reverse,name_path_reverse,df_path_reverse
#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'TIANA')
from argparse import ArgumentParser
import argparse
package_dir = os.path.dirname(os.path.abspath(__file__))

from TIANA.training import train
from TIANA.edge_weights import EdgeWeights
from TIANA.process_edge import EdgeMotif
from TIANA.make_logo import make_logo

def parse_arguments():
    """
    This is a sample script that demonstrates the use of argparse
    to parse required and optional arguments.
    """
    parser = argparse.ArgumentParser(description="TIANA script")

    # Required argument
    parser.add_argument("-o", "--output_dir",dest='output_dir',type=str,
                        required=True, help="required output directory")
    
    parser.add_argument( "--train_positive_path",dest='train_positive_path',type=str,
                        required=True, help="required path to positive training data in one hot encoding")

    parser.add_argument( "--validation_positive_path",dest='validation_positive_path',type=str,
                        required=True, help="required path to positive validation (holdout) data in one hot encoding ")

    parser.add_argument( "--trainval_positive_path",dest='trainval_positive_path',type=str,
                        required=True, help="required path to train and validation data in one hot encoding")
    
    parser.add_argument( "--neg_path",dest='neg_path',type=str,
                        required=True, help="required path to negative (background) data in one hot encoding")   
    
    parser.add_argument( "--motif_pssm_path",dest='motif_pssm_path',type=str,
                        required=True, help="required path to motif pssm file")  
    
    parser.add_argument( "--motif_threshold_path",dest='motif_threshold_path',type=str,
                        required=True, help="required path to motif threshold file")     
    
    parser.add_argument( "--tf_map",dest='tf_map',type=str,
                        required=True, help="tf group info")        
    
    # optional flag 
        
    parser.add_argument("--skip_train",dest='skip_train', 
                            action='store_true', default=False,
                            help="raise this flag if skip the training process")

    parser.add_argument("--skip_html",dest='skip_html', 
                            action='store_true', default=False,
                            help="raise this flag if skip the html generation")
    
    parser.add_argument("--skip_logo",dest='skip_logo', 
                            action='store_true', default=False,
                            help="raise this flag if skip the logo generation")
    

    args = parser.parse_args()
    return args

def step1_train(pssm,pos_train_path,pos_val_path,neg_path,model_out_path=''):
    model, history = train(pssm=pssm,
                           pos_train_path=pos_train_path,
                           pos_val_path=pos_val_path,
                           neg_path=neg_path)
    if len(model_out_path)>0:
        model.save(model_out_path)



#/home/zhl022/daima/fixed_data/TIANA_data
if __name__ == "__main__":
    package_dir = os.path.dirname(os.path.abspath(__file__))

    args = parse_arguments()
    
    # default path
    train_positive_path=args.train_positive_path
    validation_positive_path=args.validation_positive_path
    trainval_positive_path=args.trainval_positive_path
    neg_path=args.neg_path
    motif_pssm_path=args.motif_pssm_path
    motif_threshold_path=args.motif_threshold_path
    tf_map=args.tf_map
    
    skip_logo=args.skip_logo
    skip_html=args.skip_html
    skip_train=args.skip_train
    
    #output directory 
    output_dir=args.output_dir
    
    # commonly used path
    model_path = os.path.join(output_dir, 'model') #./out/model/
    npy_edge_dir =os.path.join(output_dir, 'npy_edge') #./out/npy_edge/
    result_dir =os.path.join(output_dir, 'result') #./out/result/
    logo_dir =os.path.join(result_dir,"logo_html") #./out/result/logo_html/
    result_df_path=os.path.join(result_dir, 'result_df.csv') #./out/result/result_df.csv
    pos_edge_path=os.path.join(npy_edge_dir, 'filtered_pos_edge_rank_ig.npy') #./out/npy_edge/filtered_pos_edge_rank_ig.npy
    neg_edge_path=os.path.join(npy_edge_dir, 'filtered_neg_edge_rank_ig.npy') #./out/npy_edge/filtered_pos_edge_rank_ig.npy
    
    #mkdir for npy and result dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(npy_edge_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(logo_dir, exist_ok=True)
    
    if skip_train==False:
        step1_train(pssm=motif_pssm_path,
                    pos_train_path=train_positive_path,
                    pos_val_path=validation_positive_path,
                    neg_path=neg_path,
                    model_out_path=model_path)
    
    # step 2 compute edge from model
    obj = EdgeWeights(model_path=model_path,
                  pssm=motif_pssm_path,
                  motif_cutoff_path=motif_threshold_path,
                  pos_train_path=trainval_positive_path, # use train val
                  pos_val_path=None,
                  neg_path=neg_path,
                  out_npy_dir=npy_edge_dir,
                  ncore=25,
                  neg_size=20000)
    
    obj.batch_compute_pos()
    obj.merge_pos_npy()
    obj.batch_compute_neg()
    obj.merge_neg_npy()
    obj.filter_edge_pos()
    obj.filter_edge_neg()
    
    # step 3 result generation
    motif_obj=EdgeMotif(motif_cutoff_path=motif_threshold_path,
                        pssm=motif_pssm_path,
                        pos_edge_path=pos_edge_path,
                        neg_edge_path=neg_edge_path,
                        tf_family_map=tf_map,)
    df=motif_obj.mergeRows()
    df.to_csv(result_df_path,index=False)
    
    if skip_logo==False:
        make_logo(motif_pssm_path,motif_threshold_path,logo_dir)
    
    
    if skip_html==False:
        # check if logo is empty
        logo_folder_contents = os.listdir(logo_dir)
        assert len(logo_folder_contents)>0, "logo dir is empty, make logo first"
        motif_obj.generate_html(html_dir=result_dir)


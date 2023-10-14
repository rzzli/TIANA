[![python-version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/rzzli/tnasa/issues)

# TIANA
Transcription Factors Interaction Analysis using Neural Attention (TIANA)  is a deep learning based method to derive transcription factor interactions from self-attention attributions. Through tracking the gradients inside the CNN-attention deep learning framework, TIANA enables interpertation of the transcription factor interactions that are essential in regulatory elements such as transcription factor binding sites and enhancers.

Graphical overview of the method:

<p align="center">
<img src="https://github.com/rzzli/TIANA/blob/main/image/coverFig1A.jpg"  >
</p>


### Installation
```bash
git clone https://github.com/rzzli/TIANA.git
cd TIANA
pip install -e .
```
#### optional: install in a new conda environment
```bash
conda create -n TIANA_conda python=3.8.13
conda activate TIANA_conda
git clone https://github.com/rzzli/TIANA.git
cd TIANA
pip install -e .
```

### Uninstall TIANA/conda environment
uninstall TIANA package
```bash
pip uninstall TIANA
```
remove the conda environment
```bash
conda env remove -n TIANA_conda
```

### run TIANA
```bash
python tiana.py --output_dir ./outd \
        --train_positive_path pu1_train.npy \
        --validation_positive_path pu1_val.npy \
        --trainval_positive_path pu1_trainval.npy \
        --neg_path mm10_neg200bp.npy \
        --motif_pssm_path motif_pssm.npy \
        --motif_threshold_path motif_threshold.npy \
        --tf_map tf_group.npy \
```
### Demo data can be obtained [here](http://homer.ucsd.edu/zhl022/TIANA_data/TIANA_demo.tar.gz)
### or:
```
wget http://homer.ucsd.edu/zhl022/TIANA_data/TIANA_demo.tar.gz
tar -xvf TIANA_demo.tar.gz
```
### run TIANA with pre-trained model
##### For example, in TIANA_demo folder, there is a pre trained  model file "model_pu1"
```bash
# first, make a directory called ./outd
mkdir -p ./outd
# next, copy the model folder to ./outd and rename the "model_pu1" to "model"
cp -R ./model_pu1 ./outd/
mv ./outd/model_pu1 ./outd/model

#run TIANA
python tiana.py --output_dir ./outd \
        --train_positive_path pu1_train.npy \
        --validation_positive_path pu1_val.npy \
        --trainval_positive_path pu1_trainval.npy \
        --neg_path mm10_neg200bp.npy \
        --motif_pssm_path motif_pssm.npy \
        --motif_threshold_path motif_threshold.npy \
        --tf_map tf_group.npy \
        --skip_train 
```

Full usage
```
optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        required output directory
  --train_positive_path TRAIN_POSITIVE_PATH
                        required path to positive training data in one hot encoding
  --validation_positive_path VALIDATION_POSITIVE_PATH
                        required path to positive validation (holdout) data in one hot encoding
  --trainval_positive_path TRAINVAL_POSITIVE_PATH
                        required path to train and validation data in one hot encoding
  --neg_path NEG_PATH   required path to negative (background) data in one hot encoding
  --motif_pssm_path MOTIF_PSSM_PATH
                        required path to motif pssm file
  --motif_threshold_path MOTIF_THRESHOLD_PATH
                        required path to motif threshold file
  --tf_map TF_MAP       tf group info
  --skip_train          raise this flag if skip the training process
  --skip_html           raise this flag if skip the html generation
  --skip_logo           raise this flag if skip the logo generation
```

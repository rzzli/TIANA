[![python-version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/rzzli/tnasa/issues)

# TIANA
Transcription Factors Interaction Analysis using Neural Attention (TIANA)  is a deep learning based method to derive transcription factor interactions from self-attention attributions. Through tracking the gradients inside the CNN-attention deep learning framework, TIANA enables interpertation of the transcription factor interactions that are essential in regulatory elements such as transcription factor binding sites and enhancers.

Graphical overview of the method:

<p align="center">
<img src="https://github.com/rzzli/TIANA/blob/main/image/coverFig1A.jpg" width="900" height="342">
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
        --skip_train
```

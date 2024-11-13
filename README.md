# Revisiting Drug Recommendation from a Causal Perspective

## Overview
This repository is the official implementation of [Revisiting Drug Recommendation from a Causal Perspective](https://ieeexplore.ieee.org/document/10700949
).

## Requirements
```shell
torch==1.13.0
torch_geometric==2.3.1
torch-scatter==2.1.1
torch-sparse==0.6.17
pyhealth==1.1.6
rdkit==2024.3.3
ogb==1.3.6
lightning==2.2.0
```

## Data Preparation
You can obtain the MIMIC-III dataset from [here](https://physionet.org/content/mimiciii/1.4/) and the MIMIC-IV dataset from [here](https://physionet.org/content/mimiciv/3.1/). Once downloaded, place them in the `dataset/` directory.


## Run the Code
After obtaining the datasets, you can run the model with the following script:
```shell
bash scripts/run_CRec.sh
```
or
```shell
python src/run.py --model_name CRec --dataset_name mimic3 --epochs 50 --w_pos 0.1 --w_neg 0.5 --w_reg 0.1
```
For more details, please refer to [./scrips](./scripts/) and [run.py](./src/run.py).

## Citation
If you find this work useful in your research, please cite:
```shell
@ARTICLE{10700949,
  author={Zhang, Junjie and Zang, Xuan and Chen, Hao and Yan, Xiaowei and Tang, Buzhou},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Revisiting Drug Recommendation From a Causal Perspective}, 
  year={2024},
  volume={},
  number={},
  pages={1-9},
  keywords={Drugs;Correlation;Feature extraction;Safety;MIMICs;Bioinformatics;Diseases;Vectors;Metabolism;Knowledge graphs;Causal substructure;drug recommendation;electronic health records (EHRs);molecular graphs},
  doi={10.1109/JBHI.2024.3471637}}
```

## Acknowledgement
This codebase is constructed based on the repo: [PyHealth](https://github.com/sunlabuiuc/PyHealth/tree/master). Thanks a lot for their amazing work!
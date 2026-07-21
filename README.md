# BASL
Code for ''Active Tail-Anchoring: Boundary-Aware Selective Learning for Semi-Supervised Action Quality Assessment''

This paper is currently under review.

## Environments

- RTX 3090
- CUDA: 12.4
- Python: 3.8.19
- PyTorch: 2.4.1+cu124

## Data Preperation
- The MTL-AQA dataset [[Parmar and Morris]](https://github.com/ParitoshParmar/MTL-AQA)
- The FineDiving dataset [[Xu et al.]](https://github.com/xujinglin/FineDiving)
- The JIGSAWS dataset [[Gao, Yixin, et al.]](http://cirl.lcsr.jhu.edu/jigsaws)
- The features and label files of Rhythmic Gymnastics dataset can be download from the [PAMFN](https://github.com/qinghuannn/PAMFN) repository.
- The features and label files of FineFS dataset can be download from the [FineFS](https://github.com/yanliji/FineFS-dataset) repository.

## Pretrain Model for Short-term Datasets
The Kinetics pretrained I3D downloaded from the reposity [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth)
```
model_rgb.pth
```

## Training and Evaluation
```
# train a model on MTL-AQA dataset
python run_MTL.py

# train a model on FineDiving dataset
python run_FineDiving.py

# train a model on JIGSAWS dataset
python run_JIG.py

# train a model on Rhythmic Gymnastics dataset
python run_RG.py

# train a model on FineFS dataset
python run_FineFS.py
```

## Model Weights
The model weights can be obtained [here](https://1drv.ms/f/c/056e0e22eb875f5c/IgC4-mUmPRGETIhxzwvbWDAUASe4tswGkAlnmjFOTTfxdMk?e=8fZ91L).
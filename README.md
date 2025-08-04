# DEMNet
This repository contains the codes for DEMNet: Dual-Encoder-Decoder Multi-Frame Infrared Small Target Detection Network with Motion Encoding. Code will be available after the paper is accepted.
## Requirements
- Python 3
- torch
- mmdet
- tqbm
- DCNv2
- scikit-image
- thop
- skimage
- numpy
- matplotlib
- PIL
- scipy

## Datasets

./dataset
  DAUB_DTUM
  NUDT-MIRST


## Train
python train.py --model 'WATP_DTUM_concat' --loss_func 'fullySup' --train 1 --test 0 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'DAUB_DTUM'
## Test

python train.py --model 'WATP_DTUM_concat' --loss_func 'fullySup' --train 0 --test 1 --fullySupervised True --device cuda:0 --epochs 20 --dataset 'DAUB_DTUM' --pth_path DAUB_best.pth

python train.py --model 'WATP_DTUM_concat' --loss_func 'fullySup' --train 0 --test 1 --fullySupervised True --device cuda:1 --epochs 20 --dataset 'NUDT-MIRSDT' --pth_path NUDT_best.pth

Link to the pretrained modle weights.  https://pan.baidu.com/s/15ycP9mAeGK0hx6fbeaOGrg?pwd=2494
Link to NUDT-MIRSDT dataset. https://pan.baidu.com/s/1-AxP-l75vrgm8p1nv0BWFQ?pwd=sf21





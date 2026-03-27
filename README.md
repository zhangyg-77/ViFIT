# ViFlT-assisted Histopathology: From H&E Style Standardization to Virtual Fiber Image Transformation

[ViFIT Paper](https://doi.org/10.1016/j.media.2026.104051)
Shu Wang#, Xiao Zhang#, Xingfu Wang#, Chenyong Lv, Xiahui Han, Xiong Lin, Deyong Kang, Ruolan Lin, Liwen Hu, Feng Huang*, Wenxi Liu*, Jianxin Chen*

## Pytorch implementation of ViFIT

## OS Requirements
- Linux: Ubuntu 18.04
- Python 3.7 + Pytorch 1.8.1
- NVIDIA GPU + CUDA 11.0 CuDNN 8

## Installation Guide
Install PyTorch 0.4+ and other dependencies (e.g., torchvision, visdom and dominate).
- For pip users, please type the command: pip install -r requirements.txt.
- For Conda users, please create a new Conda environment using conda env create -f environment.yml.

## Download Guide
Download the required datasets and weights:
- Public multi-style H&E datasets:
[BreaKhis](https://www.kaggle.com/datasets/ambarish/breakhis), 
[BACH](https://iciar2018-challenge.grand-challenge.org),
[BreCaHAD](https://bmcresnotes.biomedcentral.com/articles/10.1186/s13104-019-4121-7)
- H&E-to-fiber test dataset:
[H&E-to-fiber dataset](https://drive.google.com/drive/folders/1WW_PCWDytBqqndO7tZ0kzmOJ0ahzDjCN?usp=drive_link)
- Model weights used in this study:
[weights](https://drive.google.com/drive/folders/1OMvme09oOczxRaKBzHKaOffcLs_6jLrF?usp=drive_link)

## Data Preconditioning
Before training, split the large images into 256×256 pixel patches. For Stage1 with the BreaKHis dataset, modify the patch script to assign labels for multi-style images.
Run the following command to preprocess the data:
```bash
python patch.py
```
Organize the data into the following directory structure:
- Training data for SourceA: ./datasets/data/trainA
- Training data for SourceB: ./datasets/data/trainB
- Test data for SourceA: ./datasets/data/testA
- Test data for SourceB: ./datasets/data/testB
- Test results for SourceA: ./datasets/data/testC

There are two main stages in our framework
- Stage1: Style Standardization
- Stage2: Fiber Transformation
To switch between stages:
- Navigate to Stage1:
  - `cd Stage1` 
- Navigate to Stage2:
  - `cd Stage2`

## Stage1 Training and Testing
Train the model for style standardization:
```bash
python train.py --dataroot ./datasets/data --name name --cls --lab --input_nc 4 --output_nc 4
```
Test the model and save results:
```bash
python test.py --dataroot ./datasets/data --name name --cls --hebing --chutu ./datasets/data/testC --input_nc 4 --output_nc 4 --lab --shuchu 'fake_B'
```

## Stage2 Training and Testing
Train the model for fiber transformation:
```bash
python train.py --dataroot ./datasets/data --name name --Auxiliary 1
```
Test the model and save results:
```bash
python test.py --dataroot ./datasets/data --name name --cls --hebing --chutu ./datasets/data/testC --shuchu 'fake_B'
```

## Combined ViFIT Testing
To run the full pipeline (Stage1 + Stage2), use the provided script:
```bash
sh ViFIT.sh
```
Intermediate results are saved in the ./checkpoints/ directory, and the .pth files will be stored in their respective folders.

## Contact
If you have any questions, please contact Shu Wang at [shu@fzu.edu.cn](shu@fzu.edu.cn) or Wenxi Liu at [wenxiliu@hotmail.com](wenxiliu@hotmail.com).

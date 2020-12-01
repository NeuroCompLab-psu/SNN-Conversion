# Exploring the Connection Between Binary and Spiking Neural Networks

## Overview
<!--
outlines the training methodology and provides traind models for FP and binary sNN(bsnn) utilizing [BindsNet](https://github.com/BindsNET/bindsnet) for large-scale datasets, namely CIFAR-100 and ImageNet
we showed that bsnn exhibits near FP
We show that trainingSpiking Neural Networks in the binary regime results in near full precision accuracies.
we use ann to snn conver technique for training and explore a novel set of optimization for generating high acc and low latency snn 
-->
This codebase outlines a training methodology and provides trained models for Full Precision and Binary Spiking Neural Networks (B-SNNs) utilizing [BindsNet](https://github.com/BindsNET/bindsnet) for large-scale datasets, namely CIFAR-100 and ImageNet. Following the proposed procedures and design features mentioned in [our work](https://www.frontiersin.org/article/10.3389/fnins.2020.00535), we have shown that B-SNNs exhibit near full-precision accuracy even with many SNN-specific constraints. Additionally, we used ANN-SNN conversion technique for training and explored a novel set of optimizations for generating high accuracy and low latency SNNs. The optimization techniques also apply to the full precision ANN-SNN conversion.

## Requirements

- A Python installation version 3.6 or above
- The matplotlib, numpy, tqdm, and torchvision
- A PyTorch install version 1.3.0 ([pytorch.org](http://pytorch.org))
- CUDA 10.1
- The ImageNet dataset (which can be automatically downloaded by a recent version of [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet)) (If needed)

## Training from scratch
We explored various network architectures constrained by ANN-SNN conversion. The finalized network structure can be found in ```vgg.py```. Further details can be found in the [paper](http://arxiv.org/abs/2002.10064).

### Hyperparameter Settings
| Model | Batch Size | Epoch | Learning Rate | Weight Decay | Optimizer |
| ---- | ---- | ---- | ---- | ---- | ---- |
| CIFAR-100 Full Precision | 256 | 200 |  5e-2, divided by 10 at 81 and 122 epoch | 1e-4 | SGD (momentum=0.9) |
| CIFAR-100 Binary | 256 | 200 | 5e-4, halved every 30 epochs | 5e-4 (0 after 30 epochs) | Adam |
| ImageNet Full Precision| 128 | 100 |  1e-2, divided by 10 every 30 epochs | 1e-4 | SGD (momentum=0.9) |
| ImageNet Binary | 128 | 100 |  5e-4, halved every 30 epochs | 5e-4 (0 after 30 epochs) | Adam(**beta=(0.0,0.999)**) |

Note that these hyper-parameters may be further optimized.

## Evaluating Pre-trained models
We provide pre-trained models of the VGG architecture mentioned in the paper and described above, available for download. Note that the first and the last layers are not binarized for our models. The corresponding top-1 accuracies are indicated in parentheses.

* [CIFAR-100 Full Precision ANN (64.9%)](https://drive.google.com/open?id=1ZmagwfBdWVVztCdn67gmAWtfQJY3yrev)
* [CIFAR-100 Binary ANN (64.8%)](https://drive.google.com/open?id=1605x2i_noKiQ-Z4OZW9L__deR_ubvfGS)
* [ImageNet Full Precision ANN (69.05%)](https://drive.google.com/open?id=1SHXlvUrkPAkl8nQ8_LCNja5ypkqh59_x)
* [ImageNet Binary ANN (64.4%)](https://drive.google.com/open?id=12WeIAfrVNxD45NFv4HV1nSvrLa3rRZp_)

The Full Precision ANNs are trained using standard [PyTorch training practices](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and the binarization process utilizes part of the [XNOR-Net-Pytorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch) script which is the python implementation of the original [XNOR-Net](https://github.com/allenai/XNOR-Net) script.

## Running a simulation

Prepare the pre-trained model and move to the same directory, and run the following code for each model:

```python conversion.py --job-dir cifar100_test --gpu --dataset cifar100 --data . --percentile 99.9 --norm 3500 --arch vgg15ab --model bin_cifar100.pth.tar```

Full documentation of the arguments in `conversion.py`:
```
usage: conversion.py [-h] --job-dir JOB_DIR --model MODEL
                     [--results-file RESULTS_FILE] [--seed SEED] [--time TIME]
                     [--batch-size BATCH_SIZE] [--n-workers N_WORKERS]
                     [--norm NORM] [--gpu] [--one-step] [--data DATA_PATH]
                     [--arch ARCH] [--percentile PERCENTILE]
                     [--eval_size EVAL_SIZE] [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --job-dir JOB_DIR     The working directory to store results
  --model MODEL         The path to the pretrained model
  --results-file RESULTS_FILE
                        The file to store simulation result
  --seed SEED           A random seed
  --time TIME           Time steps to be simulated by the converted SNN
                        (default: 80)
  --batch-size BATCH_SIZE
                        Mini batch size
  --n-workers N_WORKERS
                        Number of data loaders
  --norm NORM           The amount of data to be normalized at once
  --gpu                 Whether to use GPU or not
  --one-step            Single step inference flag
  --data DATA_PATH      The path to ImageNet data (default: './data/)',
                        CIFAR-100 will be downloaded
  --arch ARCH           ANN architecture to be instantiated
  --percentile PERCENTILE
                        The percentile of activation in the training set to be
                        used for normalization of SNN voltage threshold
  --eval_size EVAL_SIZE
                        The amount of samples to be evaluated (default:
                        evaluate all)
  --dataset DATASET     cifar100 or imagenet
```
Depending on your computing resources, some settings can be changed to speed up or to accommodate the available device. ```--norm```, ```--batch-size```, and ```--time``` can be changed for better performance.

## Reference

If you use this code, please cite the following paper:

Sen Lu and Abhronil Sengupta. "Exploring the Connection Between Binary and Spiking Neural Networks", Frontiers in Neuroscience, Vol. 14, pp. 535 (2020).

```
@ARTICLE{10.3389/fnins.2020.00535,
AUTHOR={Lu, Sen and Sengupta, Abhronil},    
TITLE={Exploring the Connection Between Binary and Spiking Neural Networks},      	
JOURNAL={Frontiers in Neuroscience},      
VOLUME={14},      
PAGES={535},     
YEAR={2020},       
URL={https://www.frontiersin.org/article/10.3389/fnins.2020.00535},       
DOI={10.3389/fnins.2020.00535},      
ISSN={1662-453X}
}
```

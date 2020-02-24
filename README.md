# Exploring the Connection Between Binary and Spiking Neural Networks (SNN)

## Overview
<!--
outlines the training methodology adn traind model for FP and binary sNN(bsnn) utilizing [BindsNet](https://github.com/BindsNET/bindsnet) for large-scale datasets, namely CIFAR-100 and ImageNet
we showed that bsnn exhibits near FP
We show that trainingSpiking Neural Networks in the binary regime results in near full precision accuracies.
we use ann to snn conver technique for training and explore a novel set of optimization for generating high acc and low latency snn 
-->
This codebase outlines a training methodology and trained models for Binary Spiking Neural Network (BSNN) utilizing [BindsNet](https://github.com/BindsNET/bindsnet) for large-scale datasets, namely CIFAR-100 and ImageNet. Following the proposed procedures and design features mentioned in [our work](https://ismyinternetworking.com/), we have shown that the BSNN exhibits near full-precision accuracy which is significantly better than the [XNOR-Net](https://github.com/allenai/XNOR-Net) even with many SNN-specific constraints. Additionally, we used the ANN-SNN conversion techniques for training and explored a novel set of optimization for generating high accuracy and low latency SNN.

The optimization techniques also apply to the full precision ANN to SNN conversion.

## Requirements

- A Python installation version 3.6 or above
- The matplotlib, numpy, tqdm, and torchvision
- A PyTorch install version 1.3.0 ([pytorch.org](http://pytorch.org))
- CUDA 10.1
- The ImageNet dataset (which can be automatically downloaded by a recent version of [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet)) (If needed)

## Training from scratch
We explored various network structures under SNN conversion constraints and has derived a final network structure that can be found in ```vgg.py```. Further details can be found in the paper.

### Hyperparameter Settings
| Model | Batch Size | Epoch | Learning Rate | Weight Decay | Optimizer |
| ---- | ---- | ---- | ---- | ---- | ---- |
| CIFAR-100 Full Precision | 256 | 200 |  5e-2, divide by 10 at 81 and 122 epoch | 1e-4 | SGD (momentum=0.9) |
| CIFAR-100 Binary | 256 | 200 | 5e-4, halved every 30 epochs | 5e-4 (0 after 30 epochs) | Adam |
| ImageNet Full Precision| 128 | 100 |  1e-2, divide by 10 every 30 epochs | 1e-4 | SGD (momentum=0.9) |
| ImageNet Binary | 128 | 100 |  5e-4, halved every 30 epochs | 5e-4 (0 after 30 epochs) | Adam(**beta=(0.0,0.999)**) |

Note that these hyper-parameters may be further optimized.

## Evaluating Pre-trained models
We provide pre-trained models of the VGG architecture mentioned in the paper and described above, available for download. Note that the first and the last layer are not binarized for our binarized models. The corresponding top-1 accuracies are indicated in parentheses.

* [CIFAR-100 Full Precision ANN (64.9%)](https://drive.google.com/open?id=1ZmagwfBdWVVztCdn67gmAWtfQJY3yrev)
* [CIFAR-100 Binary ANN (64.8%)](https://drive.google.com/open?id=1605x2i_noKiQ-Z4OZW9L__deR_ubvfGS)
* [ImageNet Full Precision ANN (69.05%)](https://drive.google.com/open?id=1SHXlvUrkPAkl8nQ8_LCNja5ypkqh59_x)
* [ImageNet Binary ANN (64.4%)](https://drive.google.com/open?id=12WeIAfrVNxD45NFv4HV1nSvrLa3rRZp_)

The Full Precision ANNs are trained using standard [PyTorch training practices](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and the binarization process utilizes part of the [XNOR-Net-Pytorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch) script which is the python implementation of the original [XNOR-Net](https://github.com/allenai/XNOR-Net) script.

## Running a simulation

Prepare the pre-trained model and move to the same directory, and run the following code for each model:

```python main.py --job-dir test --results-file res.txt --gpu --batch-size 100 --time 170 --dataset imagenet --data D:/imagenet_data --percentile 99.9 --norm 3500 --arch vgg15ab --model ab_imgnet_bin.pth.tar```

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

## Reference

If you use this code, please cite the following paper:

Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. "Deep Clustering for Unsupervised Learning of Visual Features." Proc. ECCV (2018).

```
@InProceedings{caron2018deep,
  title={Deep Clustering for Unsupervised Learning of Visual Features},
  author={Caron, Mathilde and Bojanowski, Piotr and Joulin, Armand and Douze, Matthijs},
  booktitle={European Conference on Computer Vision},
  year={2018},
}
```

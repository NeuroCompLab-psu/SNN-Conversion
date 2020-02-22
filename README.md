# Exploring the Connection Between Binary and Spiking Networks

## Overview
This codebase provides a binary ANN to SNN conversion scheme utilizing [BindsNet](https://github.com/BindsNET/bindsnet) and a novel set of optimization proposal for large-scale datasets, namely CIFAR-100 and ImageNet. Following the proposed procedures and design features mentioned in [our work](https://ismyinternetworking.com/), the BSNN achieved near-full precision accuracy which is significantly better than the [XNOR-Net](https://github.com/allenai/XNOR-Net) even with many SNN-specific constraints. The optimization techniques also apply to the full precision ANN to SNN conversion.

## Requirements

- A Python installation version 3.6 or above
- The matplotlib, numpy, tqdm, torchvision, and the forked [BindsNet](https://github.com/BindsNET/bindsnet)
- A PyTorch install version 1.3.0 ([pytorch.org](http://pytorch.org))
- CUDA 10.1
- The ImageNet dataset (which can be automatically downloaded by a recent version of [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet)) (If needed)

## Training from scratch
Using the network structure in ```vgg.py``` under this repository, one can reproduce the same or similar accuracy. 
### Preparing Full Precision Model on CIFAR-100
| Model | Batch Size | Epoch | Learning Rate | Weight Decay | Optimizer |
| ---- | ---- | ---- | ---- | ---- | ---- |
| CIFAR-100 Full Precision | 256 | 200 |  5e-2, divide by 10 at 81 and 122 epoch | 1e-4 | SGD (momentum=0.9) |
| CIFAR-100 Binary | 256 | 200 | 5e-4, halved every 30 epochs | 1e-4 (0 after 30 epochs) | Adam |
| ImageNet Full Precision| 128 | 100 |  1e-2, divide by 10 every 30 epochs | 1e-4 | SGD (momentum=0.9) |
| ImageNet Binary | 128 | 100 |  5e-4, halved every 30 epochs | 5e-4 (0 after 30 epochs) | Adam(**beta=(0.0,0.999)**) |

Note that these hyper-parameters may be further optimized.

## Evaluating Pre-trained models
We provide pre-trained models of the VGG architecture mentioned in the paper and described above, available for download. Note that the first and the last layer are not binarized for our binarized models. The corresponding op-1 accuracies are indicated in parentheses.

* [CIFAR-100 Full Precision ANN (64.9%)](https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar)
* [CIFAR-100 Binary ANN(64.8%)](https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar)
* [ImageNet Full Precision ANN(69.05%)](https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar)
* [ImageNet Binary ANN(64.4%)](https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar)

The Full Precision ANNs are trained using standard [PyTorch training practices](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and the binarization process utilizes part of the [XNOR-Net-Pytorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch) script which is the python implementation of the original [XNOR-Net](https://github.com/allenai/XNOR-Net).

## Running a simulation

Prepare the pre-trained model and move to the same directory, and run the following code for each model:



Full documentation of the unsupervised training code `conversion.py`:
```
usage: conversion.py [-h] --job-dir JOB_DIR --model MODEL
                     [--results-file RESULTS_FILE] [--seed SEED] [--time TIME]
                     [--batch-size BATCH_SIZE] [--n-workers N_WORKERS]
                     [--norm NORM] [--gpu] [--one-step] [--data DATA_PATH]
                     [--arch ARCH] [--percentile PERCENTILE]
                     [--eval_size EVAL_SIZE] [--dataset DATASET]

required arguments:
  --job-dir JOB_DIR     The working directory to store results
  --model MODEL         The path to the pre-trained model

optional arguments:
  -h, --help            show this help message and exit
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
  --one-step            Single-step inference flag
  --data DATA_PATH      The path to ImageNet data (default: './data/)',
                        CIFAR-100 will be downloaded
  --arch ARCH           ANN architecture to be instantiated
  --percentile PERCENTILE
                        The percentile of activation in the training set to be
                        used for normalization of SNN voltage threshold
  --eval_size EVAL_SIZE
                        The number of samples to be evaluated
  --dataset DATASET     cifar100 or imagenet
```


## License

You may find out more about the license [here](https://github.com/facebookresearch/deepcluster/blob/master/LICENSE).

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

# ANN to SNN conversion using BindsNet

## Conversion
This code modifies on top of the [mini-batch processing code](https://github.com/BINDS-LAB-UMASS/snn-minibatch/tree/master/minibatch/conversion) as described in the paper [Minibatch Processing in Spiking Neural Networks](https://arxiv.org/abs/1909.02549). Our work use this framework to optimize the latency aspect of ANN-SNN conversion process.

Moreover, we provide the following features for better analysis:
* Adaptation to CIFAR-100 and ImageNet dataset
* VGG15 model structure as described in our paper
* Accuracy vs timesteps plot over time


## Requirements

- a Python installation version 3.6 or above
- the matplotlib, numpy, tqdm, torchvision, and the forked [BindsNet](https://github.com/BindsNET/bindsnet)
- a PyTorch install version 1.3.0 ([pytorch.org](http://pytorch.org))
- CUDA 10.1
- The ImageNet dataset (which can be automatically downloaded by recent version of [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet)) (If needed)

## Pre-trained models
We provide pre-trained model of the VGG-15 architecture mentioned in the paper as well as the binarized version of it, available for download.
* [CIFAR-100 Full Precision ANN](https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar)
* [CIFAR-100 Binary ANN](https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar)
* [ImageNet Full Precision ANN](https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar)
* [ImageNet Binary ANN](https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar)

The Full Precision ANNs are trained using standard [PyTorch training scripts](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) where as the binarization process was done with [XNOR-Net-Pytorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch) which is the python implementation of the [XNOR-Net](https://github.com/allenai/XNOR-Net).



## Running a simulation

Copy the pre-trained model to the same directory, and run the following code


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
  --model MODEL         The path to the pretrained model

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
  --one-step            Single step inference flag
  --data DATA_PATH      The path to ImageNet data (default: './data/)',
                        CIFAR-100 will be downloaded
  --arch ARCH           ANN architecture to be instantiated
  --percentile PERCENTILE
                        The percentile of activation in the training set to be
                        used for normalization of SNN voltage threshold
  --eval_size EVAL_SIZE
                        The amount of samples to be evaluated
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

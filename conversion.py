import argparse
import os
from time import time
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from bindsnet.conversion import ann_to_snn
from bindsnet.encoding import RepeatEncoder
from bindsnet.datasets import ImageNet, CIFAR100, DataLoader
import torchvision.transforms as transforms
from vgg import vgg_15_avg_before_relu


def main(args):
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_workers == -1:
        args.n_workers = args.gpu * 4 * torch.cuda.device_count()

    device = torch.device("cuda" if args.gpu else "cpu")

    # Load trained ANN from disk.
    if args.arch == 'vgg15ab':
        ann = vgg_15_avg_before_relu(dataset=args.dataset)
    # add other architectures here#
    else:
        raise ValueError('Unknown architecture')


    ann.features = torch.nn.DataParallel(ann.features)
    ann.cuda()
    if not os.path.isdir(args.job_dir):
        os.mkdir(args.job_dir)
    f = os.path.join('.', args.model)
    try:
        dictionary = torch.load(f=f)['state_dict']
    except KeyError:
        dictionary = torch.load(f=f)
    ann.load_state_dict(state_dict=dictionary, strict=True)

    if args.dataset=='imagenet':
        input_shape=(3,224,224)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # the actual data to be evaluated
        val_loader = ImageNet(
            image_encoder=RepeatEncoder(time=args.time, dt=1.0),
            label_encoder=None,
            root=args.data,
            download=False,
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            split='val')
        # a wrapper class
        dataloader = DataLoader(
            val_loader,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=args.gpu,
        )
        # A loader of samples for normalization of the SNN from the training set
        norm_loader = ImageNet(
            image_encoder=RepeatEncoder(time=args.time, dt=1.0),
            label_encoder=None,
            root=args.data,
            download=False,
            split='train',
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize, ]
            ),
        )

    elif args.dataset == 'cifar100':
        input_shape=(3, 32, 32)
        print('==> Using Pytorch CIFAR-100 Dataset')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                         std=[0.267, 0.256, 0.276])
        val_loader = CIFAR100(
            image_encoder=RepeatEncoder(time=args.time, dt=1.0),
            label_encoder=None,
            root=args.data,
            download=True,
            train=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize, ]
            )
        )

        dataloader = DataLoader(
            val_loader,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=args.gpu,
        )

        norm_loader = CIFAR100(
            image_encoder=RepeatEncoder(time=args.time, dt=1.0),
            label_encoder=None,
            root=args.data,
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize, ]
            )
        )
    else:
        raise ValueError('Unsupported dataset.')

    if args.eval_size == -1:
        args.eval_size = len(val_loader)

    for step, batch in enumerate(torch.utils.data.DataLoader(norm_loader, batch_size=args.norm)):
        data = batch['image']
        break

    snn = ann_to_snn(ann, input_shape=input_shape, data=data, percentile=args.percentile)


    torch.cuda.empty_cache()
    snn = snn.to(device)

    correct = 0
    t0 = time()
    accuracies = np.zeros((args.time, (args.eval_size//args.batch_size)+1), dtype=np.float32)
    for step, batch in enumerate(tqdm(dataloader)):
        if (step+1)*args.batch_size > args.eval_size:
            break
        # Prep next input batch.
        inputs = batch["encoded_image"]
        labels = batch["label"]
        inpts = {"Input": inputs}
        if args.gpu:
            inpts = {k: v.cuda() for k, v in inpts.items()}

        snn.run(inpts=inpts, time=args.time, step=step, acc= accuracies, labels=labels,one_step=args.one_step)
        output_voltages = snn.layers['44'].summed
        prediction = torch.softmax(output_voltages, dim=1).argmax(dim=1)
        correct += (prediction.cpu() == labels).sum().item()
        snn.reset_()
    t1 = time() - t0

    final = accuracies.sum(axis=1) / args.eval_size

    plt.plot(final)
    plt.suptitle('{} {} ANN-SNN@{} percentile'.format(args.dataset, args.arch, args.percentile), fontsize=20)
    plt.xlabel('Timestep', fontsize=19)
    plt.ylabel('Accuracy', fontsize=19)
    plt.grid()
    plt.show()
    plt.savefig('{}/{}_{}.png'.format(args.job_dir, args.arch, args.percentile))
    np.save('{}/voltage_accuracy_{}_{}.npy'.format(args.job_dir, args.arch, args.percentile), final)


    accuracy = 100 * correct / args.eval_size

    print(f"SNN accuracy: {accuracy:.2f}")
    print(f"Clock time used: {t1:.4f} ms.")
    path = os.path.join(args.job_dir, "results", args.results_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        with open(path, "w") as f:
            f.write("seed,simulation time,batch size,inference time,accuracy\n")
    to_write = [args.seed, args.time, args.batch_size, t1, accuracy]
    to_write = ",".join(map(str, to_write)) + "\n"
    with open(path, "a") as f:
        f.write(to_write)

    return t1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, required=True, help='The working directory to store results')
    parser.add_argument("--model", type=str, required=True, help='The path to the pretrained model')
    parser.add_argument("--results-file", type=str, default='sim_result.txt', help='The file to store simulation result')
    parser.add_argument("--seed", type=int, default=0, help='A random seed')
    parser.add_argument("--time", type=int, default=80, help='Time steps to be simulated by the converted SNN (default: 80)')
    parser.add_argument("--batch-size", type=int, default=100, help='Mini batch size')
    parser.add_argument("--n-workers", type=int, default=4, help='Number of data loaders')
    parser.add_argument("--norm", type=int, default=128, help='The amount of data to be normalized at once')
    parser.add_argument("--gpu", action="store_true", help='Whether to use GPU or not')
    parser.add_argument("--one-step", action="store_true", help='Single step inference flag')
    parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
                        help='The path to ImageNet data (default: \'./data/)\', CIFAR-100 will be downloaded')
    parser.add_argument("--arch", type=str, default='vgg15ab', help='ANN architecture to be instantiated')
    parser.add_argument("--percentile", type=float, default=99.7, help='The percentile of activation in the training set to be used for normalization of SNN voltage threshold')
    parser.add_argument("--eval_size", type=int, default=-1, help='The amount of samples to be evaluated (default: evaluate all)')
    parser.add_argument("--dataset", type=str, default='cifar100', help='cifar100 or imagenet')

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

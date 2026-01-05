
import os
import sys

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_utils import RandAugment

#########################

def getCIFAR10(datadir, size, batchsize=64, augment=False):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if augment:  
        N = 2
        M = 14
        transform_train.transforms.insert(0, RandAugment(N, M))

    # Download and create datasets
    print(f'-I({__file__}): Loading CIFAR10 dataset...')

    trainset = datasets.CIFAR10(root=datadir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=8)

    testset = datasets.CIFAR10(root=datadir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Return
    print(f'-I({__file__}): CIFAR10 loaded')

    return (trainloader, testloader, classes)

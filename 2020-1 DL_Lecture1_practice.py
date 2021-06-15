# import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# from google.colab import drive
# drive.mount('/gdrive')
source_dir = '/gdrive/My Drive/directory/CIFAR10'

def Lecture1_practice():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    imageloader = data_loader()
    dataiter = iter(imageloader)
    images, labels = dataiter.next()
    images = im_sum(images)
    imshow(torchvision.utils.make_grid(images))

    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def data_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = datasets.CIFAR10(root=source_dir, train=True, download=True, transform=transform)
    trainset = datasets.CIFAR10(root=source_dir, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    return trainloader


def imshow(img): # functions to show an image
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def im_sum(img):
    tmp = img
    for i in range(4):
        if i != 0:
            for j in range(i):
                img[i] += tmp[j]
            img[i] = img[i] / (i + 1)
    return img

if __name__ == '__main__':
    data_loader()

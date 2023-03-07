from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import datasets, transforms
from Neural import Net

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
model = torch.load('convolutional_test_model')


def test(mod, dev, loader):
    mod.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(dev), target.to(dev)
            out = mod(data)
            test_loss += F.nll_loss(out, target, reduction='sum').item()  # sum up batch loss
            final = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += final.eq(target.view_as(final)).sum().item()
            test_loss /= len(loader.dataset)
            print('\nTest skup: Srednji gubitak: {:.4f}, Preciznost: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, len(loader.dataset),
                100. * correct / len(loader.dataset)))


kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
batch_size = 128
dataL = torch.utils.data.DataLoader(
    datasets.MNIST('./data',download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

#test(model, device, dataL)
img = cv2.imread('digits/testpicture.png', 0)
img = img.reshape(1, 1, 28, 28)
img = np.float32(img)
img = ((img / 255) - 0.1307) / 0.3081
img = torch.from_numpy(img)
model.eval()
with torch.no_grad():
    output = model(img)
    prediction = output.argmax(dim=1, keepdim=True)
    prediction = int(prediction)
    print("To je " + str(prediction))

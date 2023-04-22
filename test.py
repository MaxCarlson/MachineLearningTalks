import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batchSize = 32

mnistDataset = datasets.MNIST('./data', train=True, download=True, 
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ]))

mnistExample = DataLoader(mnistDataset, batch_size=1)
batchIdx, (exampleInput, exampleTarget) = next(enumerate(mnistExample))
x = exampleInput.item()
plt.imshow(x)
plt.show()
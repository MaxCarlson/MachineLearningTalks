#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from grapher import Grapher
from NeuralNetworks import OneLayerNN, TwoLayerNN

#output1 =  [[1,0],[0,1],[0,1],[0,1]]
#output2 =  [[1,0],[1,0],[0,1],[1,0]]
#output3 =  [[1,0],[0,1],[1,0],[0,1]]

device = ("cpu")

def train(epochs, dataloader, model, lossFN, optimizer, grapher):
    size = len(dataloader.dataset)
    model.train()
    printInterval = int(epochs / 10)

    for e in range(epochs):
        if e % printInterval == 0:
            grapher.addOutput(model, training=True)

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = lossFN(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, lossFN, grapher):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += lossFN(pred, y).item()
            correct += ((pred > 0.5) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    grapher.addOutput(model)  



def trainModel(ModelType, outputVar):
    model = ModelType().to(device)
    x = torch.Tensor(inputVar)
    y1 = torch.Tensor(outputVar)

    lossFN = torch.nn.MSELoss()
    #optim = torch.optim.SGD(model.parameters(), lr=1)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    batchSize = 4
    epochs = 1000
    dataset1 = TensorDataset(x, y1)
    dataLoaderTrain = DataLoader(dataset1, batch_size=batchSize, shuffle=True)
    dataLoaderTest = DataLoader(dataset1, batch_size=batchSize, shuffle=False)

    grapher = Grapher()
    train(epochs, dataLoaderTrain, model, lossFN, optim, grapher)
    test(dataLoaderTest, model, lossFN, grapher)
    grapher.graph()

    model.eval()
    t1 = model(torch.Tensor([0,0]))
    t2 = model(torch.Tensor([0,1]))
    t3 = model(torch.Tensor([1,1]))
    t4 = model(torch.Tensor([1,0]))
    ipn = 5



inputVar = [[0,0],[0,1],[1,1],[0,1]]
output1 = [[0], [1], [1], [1]] # OR
output2 = [[0], [0], [1], [0]] # AND
output3 = [[1], [0], [0], [0]] # ~AND
output4 = [[1], [1], [0], [1]] # NOR
output5 = [[0], [1], [0], [1]] # XOR


trainModel(OneLayerNN, output1)
#trainModel(OneLayerNN, output2)
#trainModel(OneLayerNN, output3)
#trainModel(OneLayerNN, output4)
#trainModel(OneLayerNN, output5)

trainModel(TwoLayerNN, output5)

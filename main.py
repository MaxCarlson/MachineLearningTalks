#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

inputVar = [[0,0],[0,1],[1,1],[0,1]]
output1 = [[0], [1], [1], [1]]
output2 = [[0], [0], [1], [0]]
output3 = [[0], [1], [0], [1]]

#output1 =  [[1,0],[0,1],[0,1],[0,1]]
#output2 =  [[1,0],[1,0],[0,1],[1,0]]
#output3 =  [[1,0],[0,1],[1,0],[0,1]]

x = torch.Tensor(inputVar)
y1 = torch.Tensor(output1)

batchSize = 2
dataset1 = TensorDataset(x, y1)
dataLoader1 = DataLoader(dataset1, batch_size=batchSize, shuffle=False)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            #nn.Sigmoid(),
        )
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.model(x)
        return logits

device = ("cpu")
model1 = NN().to(device)
#print(model1)

class Grapher:
    def __init__(self):
        self.outputs = []

    def addOutput(self, model, training=False):
        model.eval()
        inputs = np.mgrid[0:1:10j, 0:1:10j]
        #print(inputs)
        outputs = np.zeros((10,10))
        #print(outputs)

        for x in range(10):
            for y in range(10):
                out = model(torch.Tensor([inputs[0,x,y], inputs[1,x,y]])).item()
                outputs[x,y] = out

        #print(outputs)
        self.outputs.append(outputs)
        if training:
            model.train()

    def graph(self):
        cols = 4
        rows = int(len(self.outputs) / 4)
        fig = plt.figure(figsize=(cols, rows))
        for i in range(len(self.outputs)):
            fig.add_subplot(rows, cols, i+1)
            plt.imshow(self.outputs[i], cmap='hot', interpolation='nearest', origin='lower')
        plt.show()

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
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    #correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    grapher.addOutput(model)  


lossFN = torch.nn.MSELoss()
optim = torch.optim.SGD(model1.parameters(), lr=0.5)

grapher = Grapher()

train(15, dataLoader1, model1, lossFN, optim, grapher)
test(dataLoader1, model1, lossFN, grapher)

grapher.graph()
import numpy as np
import torch
import matplotlib.pyplot as plt

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
        rows = int(len(self.outputs) / 4) + 1
        fig = plt.figure(figsize=(cols, rows))
        for i in range(len(self.outputs)):
            fig.add_subplot(rows, cols, i+1)
            plt.imshow(self.outputs[i], cmap='hot', interpolation='nearest', origin='lower')
        plt.show()
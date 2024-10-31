# Introduction to Machine Learning with Neural Networks

This repository contains a Jupyter Notebook **IntroToML.ipynb**, **Introduction to Machine Learning**, that provides an overview of fundamental machine learning concepts, particularly neural networks. The notebook includes explanations, code examples, and visualizations, ideal for beginners exploring machine learning and deep learning.

## Overview

### What Makes ML Interesting
Machine learning (ML) stands out because:
- Traditional programming requires explicit rules, but ML models learn from data.
- Tasks such as image recognition and captioning are more approachable with ML, as models start with an essentially random state and improve over time.

### Machine Learning Paradigms
The notebook covers the primary paradigms in machine learning:
- **Supervised Learning**: Predicts outputs based on labeled input data.
- **Unsupervised Learning**: Finds patterns in data without labeled outputs.
- **Reinforcement Learning**: Trains agents to take actions that maximize cumulative rewards in an environment.

## Key Concepts

### Inputs and Targets
- **Inputs**: Data features provided to the model, often represented in tabular form with each row as a sample.
- **Targets**: Desired outputs that the model aims to predict, typically stored in the last column of a dataset.

Example:
| Height | Weight | Gender | Age | Smokes | Heart Disease Probability |
|--------|--------|--------|-----|--------|----------------------------|
| 71     | 165    | 1      | 27  | 1      | 0.170                      |
| 64     | 137    | 0      | 41  | 0      | 0.053                      |

### Dataset Symbols
- **N**: Number of samples
- **M**: Number of features per sample
- **x<sub>i</sub>**: Individual input vector (of length M)
- **y<sub>i</sub>**: Target output for each sample
- **ŷ<sub>i</sub>**: Predicted output by the model for each input x<sub>i</sub>
- **φ**: Represents the Activation Function
- **L<sub>ce</sub>**: Cross Entropy Loss 

### Representations
The model learns a function `f(x) = y`, attempting to approximate target values `y` for unseen data points by minimizing the error between predictions `ŷ` and actual targets `y`.

### Classification vs. Regression
- **Regression**: Predicts continuous outputs (e.g., predicting price).
- **Classification**: Predicts discrete labels (e.g., identifying if an image is a cat or dog). For classification tasks, targets are often encoded as **one-hot vectors**.

### Neurons and Activation Functions
- **Neurons**: Basic units in neural networks, performing weighted sums of inputs and passing results through activation functions.
- **Activation Functions**:
  - **Sigmoid**: φ(z) = 1 / (1 + e^-z)
  - **ReLU**: φ(z) = max(0, z)
  - **Linear**: Rarely used in hidden layers as it limits the network's complexity.

### Feed-Forward Neural Network Structure
A typical feed-forward network consists of layers of neurons:
- Each neuron’s output in one layer is connected to neurons in the next layer.
- Layers are organized sequentially, with inputs flowing forward to produce an output.

### Training Process
1. **Initialization**: Networks start with random weights.
2. **Forward Pass**: Outputs are calculated based on inputs.
3. **Error/Loss Calculation**:
   - **Mean Absolute Error (MAE)**: MAE = (1 / N) Σ |y - ŷ|
   - **Mean Squared Error (MSE)**: MSE = (1 / N) Σ (y - ŷ)^2
   - **Cross-Entropy Loss** for classification tasks: L<sub>CE</sub> = -Σ t<sub>i</sub> log(p<sub>i</sub>)
4. **Backpropagation**: Gradients are calculated to update weights in the direction that minimizes error.

### Gradient Descent and Variants
- **Gradient Descent**: Optimization algorithm to minimize the loss function by adjusting weights.
- **Stochastic Gradient Descent (SGD)**: Updates weights based on a subset (mini-batch) of data, making it computationally efficient for large datasets.

## Example Usage

The notebook includes code to train a neural network on the MNIST dataset. Key sections cover:
- Data loading and preprocessing with `torchvision`.
- Network architecture definition using PyTorch's `nn.Module`.
- Training and testing loops, including forward pass, loss computation, and optimization steps.

### Code Sample
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return F.log_softmax(self.l3(x), dim=1)

network = Net()
print(network)  
```

### Results

Training results include accuracy and loss metrics on test data after each epoch. Example output:

```
Test set: Avg. loss: 0.3261, Accuracy: 54380/60000 (91%)
```


Running the Notebook

1. Clone the repository:
```
git clone https://github.com/YourUsername/IntroToML.git
cd IntroToML
```

2. Launch Jupyter Notebook and open IntroToML.ipynb:
```
jupyter notebook
```

Explore the notebook for a comprehensive introduction to neural networks and hands-on experience with ML concepts.

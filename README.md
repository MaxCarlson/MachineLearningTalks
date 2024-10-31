# Intro to Machine Learning

This repository provides an introduction to essential machine learning concepts with a focus on neural networks, featuring a Jupyter notebook that combines theoretical explanations with practical examples. Using Python and PyTorch, it covers key ML principles, data representation, network structure, training mechanisms, and evaluation.

## Notebook Overview

The notebook includes:

### 1. **Introduction to Machine Learning**
   - An overview of what makes machine learning distinct from traditional programming.
   - Explanation of different ML paradigms: Supervised, Unsupervised, and Reinforcement Learning.

### 2. **Data and Targets**
   - Structure and format of input data for machine learning.
   - Overview of target variables, including examples of datasets used in classification tasks (e.g., MNIST).

### 3. **Key Concepts and Symbols**
   - Definitions of basic ML symbols and notations, such as number of samples (N), features (M), input vectors, and model output.
   - Explanation of classification versus regression tasks, including encoding methods like one-hot encoding.

### 4. **Neural Networks and Activation Functions**
   - Structure of a neural network and neuron operations.
   - Common activation functions (Sigmoid, ReLU) and their significance in network performance.

### 5. **Network Structure and Forward Pass**
   - Detailed breakdown of a fully connected feed-forward neural network and the forward pass mechanism.
   - Code examples for implementing a simple neural network using PyTorch.

### 6. **Training and Loss Functions**
   - Explanation of training processes, including weight updates and error backpropagation.
   - Overview of loss functions (e.g., Mean Absolute Error, Mean Squared Error, Cross-Entropy) and their role in optimization.

### 7. **Gradient Descent and Optimization**
   - Introduction to gradient descent and stochastic gradient descent (SGD).
   - Visualizations and explanations on using gradients to minimize error during training.

### 8. **Testing and Evaluation**
   - Code to test model accuracy and loss on validation data.
   - Methods for evaluating model performance with examples of training output.

## Requirements

- **Python** 3.6+
- **PyTorch** for neural network implementation
- **Matplotlib** for data visualization

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/IntroToML.git
   cd IntroToML
   ```
2. Install requirements 
   ```bash
   pip install -r requirements.txt
   ```
3. Run Notebook
   ```bash
   jupyter notebook IntroToML.ipynb
   ```

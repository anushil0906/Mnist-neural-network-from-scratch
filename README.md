**Description**

This project implements a fully connected neural network from scratch using NumPy
to classify handwritten digits from the MNIST dataset. The goal was to gain a deep
understanding of neural network internals, including forward propagation, backpropagation,
gradient descent, and activation functions without relying on deep learning frameworks.


**Architecture**

Network Architecture:
- Input Layer: 784 neurons (28×28 image pixels)
- Hidden Layer 1: 72 neurons (ReLU)
- Hidden Layer 2: 36 neurons (ReLU)
- Output Layer: 10 neurons (Softmax)

**Requirement**
NumPy: For matrix operations 
Pandas: For loading the dataset 
Matplotlib: For visualizing digits

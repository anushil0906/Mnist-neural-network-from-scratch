**Description**

This project implements a fully connected neural network from scratch using NumPy
to classify handwritten digits from the MNIST dataset. The goal was to gain a deep
understanding of neural network internals, including forward propagation, backpropagation,
gradient descent, and activation functions without relying on deep learning frameworks.

**Requirement**

NumPy: For matrix operations 

Pandas: For loading the dataset 

Matplotlib: For visualizing digits

**Architecture**

Network Architecture:
- Input Layer: 784 neurons (28×28 image pixels)
- Hidden Layer 1: 72 neurons (ReLU)
- Hidden Layer 2: 36 neurons (ReLU)
- Output Layer: 10 neurons (Softmax)

**Key Concept Implemented** 

- Weight and bias initialization
- Forward propagation
- ReLU and Softmax activation functions
- Cross-entropy loss
- Backpropagation using chain rule
- Gradient descent optimization
- One-hot encoding of labels
- Model evaluation using accuracy

**MATH USED**
The model uses softmax with cross-entropy loss at the output layer.

∂
𝐿
∂
𝑍
(
3
)
=
𝐴
(
3
)
−
𝑌
∂Z
(3)
∂L
	​

=A
(3)
−Y

Where:

𝐴
(
3
)
A
(3)
 = predicted probabilities

𝑌
Y = one-hot encoded true labels

This gradient represents the error signal at the output layer.



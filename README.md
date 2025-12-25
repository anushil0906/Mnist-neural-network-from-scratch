**DEMO**

https://github.com/user-attachments/assets/d404ad0e-e02d-4435-96a5-4d66ce643939


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

**Results**

After training for 500 iterations with a learning rate of 0.2, the network achieves around 92% accuracy on the test set.

**MATH USED**

**1 Forward Propagation**


Z₁ = W₁X + b₁
A₁ = ReLU(Z₁)

Z₂ = W₂A₁ + b₂
A₂ = ReLU(Z₂)

Z₃ = W₃A₂ + b₃
A₃ = Softmax(Z₃)


**2 Loss Function**

![WhatsApp Image 2025-12-25 at 5 09 57 PM](https://github.com/user-attachments/assets/42a9261e-99dd-40ad-911d-ba903f7345ba)


**3 Backpropagation**

Backpropagation is the core learning mechanism used to train the neural network.
It computes how much each weight contributes to the final prediction error and updates the weights accordingly.


**Loss Gradient w.r.t Weights**

![WhatsApp Image 2025-12-25 at 4 57 53 PM](https://github.com/user-attachments/assets/ce99e37b-1cb1-4d6b-869a-19b34f0370e5)


**Chain Rule in Backpropagation**

![WhatsApp Image 2025-12-25 at 4 58 26 PM](https://github.com/user-attachments/assets/abdee58f-32cf-4faa-8280-4fcdd8c32421)


**Weight Updates via Gradient Descent**

![WhatsApp Image 2025-12-25 at 4 53 46 PM](https://github.com/user-attachments/assets/bb5d1ddc-6b36-4cf9-88cf-548f01c65c5f)












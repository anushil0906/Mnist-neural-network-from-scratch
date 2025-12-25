#importing necessary libraries for numerical operations, data manipulation and plotting images
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt        
# importing the dataset from CSV file
data = pd.read_csv(r'C:\Users\sharm\OneDrive\Documents\GitHub\Mnist-neural-network-from-scratch\train\train.csv')

# Converting the DataFrame to a NumPy array for easier manipulation
data = np.array(data)
# Getting the number of samples (m) and features (n)
m, n = data.shape
# Shuffling the data to ensure randomness in training
np.random.shuffle(data)
# Splitting the data into development and training sets
data_dev = data[0:1000].T         # Development set of 1000 samples
Y_dev = data_dev[0]               # Labels for the development set
X_dev = data_dev[1:n]             # Features for the development set
X_dev = X_dev / 255               # Normalizing the input data

data_train = data[1000:m].T       # Training set
Y_train = data_train[0]           # Labels for the training set
X_train = data_train[1:n]         # Features for the training set
X_train = X_train / 255           # Normalizing the input data
def initialize_parameters():
    """Initializes the parameters (weights and biases) for the neural network."""
    # Weight matrices are initialized randomly with values between -0.5 and 0.5
    W1 = np.random.rand(72, 784) - 0.5  # Weights for the first layer (64 neurons, 784 inputs)
    b1 = np.random.rand(72, 1) - 0.5    # Bias for the first layer
    W2 = np.random.rand(36, 72) - 0.5    # Weights for the second layer (32 neurons, 64 inputs)
    b2 = np.random.rand(36, 1) - 0.5    # Bias for the second layer
    W3 = np.random.rand(10, 36) - 0.5    # Weights for the output layer (10 neurons, 32 inputs)
    b3 = np.random.rand(10, 1) - 0.5    # Bias for the output layer
    return W1, b1, W2, b2, W3, b3



def ReLU(Z):
  """Applies the ReLU activation function."""
  return np.maximum(Z, 0)

def softmax(Z):
  """Applies the softmax activation function."""
  A = np.exp(Z) / sum(np.exp(Z))
  return A


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    """Performs forward propagation through the neural network."""
    Z1 = W1.dot(X) + b1  # Linear transformation for the first layer
    A1 = ReLU(Z1)        # Activation function for the first layer
    Z2 = W2.dot(A1) + b2  # Linear transformation for the second layer
    A2 = ReLU(Z2)        # Activation function for the second layer
    Z3 = W3.dot(A2) + b3  # Linear transformation for the output layer
    A3 = softmax(Z3)     # Applying softmax to get output probabilities
    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):
  """Calculates the derivative of the ReLU activation function."""
  return Z > 0


def one_hot(Y):
    """Converts the labels to one-hot encoding."""
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # Initialize a zero array
    one_hot_Y[np.arange(Y.size), Y] = 1           # Set the correct indices to 1
    one_hot_Y = one_hot_Y.T                       # Transpose the array for correct shape
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    """Performs backward propagation to compute gradients."""
    one_hot_Y = one_hot(Y)  # Convert labels to one-hot encoding
    dZ3 = A3 - one_hot_Y     # Compute the gradient for the output layer
    dW3 = 1 / m * dZ3.dot(A2.T)  # Gradient for weights of output layer
    db3 = 1 / m * np.sum(dZ3)     # Gradient for biases of output layer

    # Backpropagate through the second layer
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)  # Gradient for weights of second layer
    db2 = 1 / m * np.sum(dZ2)     # Gradient for biases of second layer

    # Backpropagate through the first layer
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)    # Gradient for weights of first layer
    db1 = 1 / m * np.sum(dZ1)      # Gradient for biases of first layer

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    """Updates the parameters using gradient descent."""
    W1 = W1 - alpha * dW1  # Update weights of the first layer
    b1 = b1 - alpha * db1   # Update biases of the first layer
    W2 = W2 - alpha * dW2   # Update weights of the second layer
    b2 = b2 - alpha * db2   # Update biases of the second layer
    W3 = W3 - alpha * dW3   # Update weights of the output layer
    b3 = b3 - alpha * db3   # Update biases of the output layer
    return W1, b1, W2, b2, W3, b3
def get_predictions(A3):
    """Returns the predicted classes based on the softmax output."""
    return np.argmax(A3, 0)  # Returns the index of the maximum probability for each sample


def get_accuracy(predictions, Y):
    """Calculates the accuracy of the model."""
    print(predictions, Y)  # Optional: Print predictions and true labels for inspection
    return np.sum(predictions == Y) / Y.size  # Calculate the ratio of correct predictions


def gradient_descent(X, Y, alpha, iterations):
    """Trains the neural network using gradient descent."""
    W1, b1, W2, b2, W3, b3 = initialize_parameters()  # Initialize parameters
    for i in range(1, iterations+1):
        # Forward propagation
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)

        # Backward propagation
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)

        # Update parameters
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        # Print the progress every 50 iterations
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print("Accuracy: ", get_accuracy(predictions, Y))

    return W1, b1, W2, b2, W3, b3
# Training the neural network on the training set
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.2, 500)     

def make_predictions(X, W1, b1, W2, b2, W3, b3):
  """Generates predictions for the input data using the trained network."""
  _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
  predictions = get_predictions(A3)
  return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    """Tests the model's prediction for a specific image index."""
    current_image = X_train[:, index, None]  # Get the current image (column vector)
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)  # Get prediction for the current image
    label = Y_train[index]  # Get the true label for the current image
    print("Prediction: ", prediction)  # Print the predicted class
    print("Label: ", label)  # Print the true label

    # Reshape the image for displaying and scale back to 0-255
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()  # Set the colormap to gray
    plt.imshow(current_image, interpolation='nearest')  # Display the image
    plt.show()  # Show the plot

indexes = [1,50,100,150,200,250,300,350,400,450]
for i in indexes:
  test_prediction(i, W1, b1, W2, b2, W3, b3)    
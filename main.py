import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train -> (60000, 28, 28) -> 60000 images of 28x28 pixels [3D Matrix]
#y_train -> (60000,) -> 60000 labels (0-9)
#X_test -> (10000, 28, 28) 
#y_test -> (10000,)
X_train = X_train / 255.0 #Originally 0-255, normalise to 0-1 range 
X_train = X_train.reshape(-1, 784) #Flatten to 784 neurons -> (60000, 784) [2D Matrix]

X_test = X_test / 255.0
X_test = X_test.reshape(-1, 784)

#One-hot encoding is needed to remove the ordinal relationship between the classes (0-9) and to allow the network to output probabilities for each class.
#It also makes it easier to compare probabilites output by the network with the true labels during training using categorical cross-entropy loss.
#ie. Y and A2 have the same shape (60000, 10) and we can directly compare them to calculate loss and gradients.
def one_hot(y):
    one_hot = np.zeros((y.size, 10)) #Make a 2D array of size y.size x 10, fill with zeros
    one_hot[np.arange(y.size), y] = 1 #Vectorized indexing -> putting a 1 in each row at the index of the correct class
    #Does this by setting np.arange(y.size) as the row indices and y as the column indices, so for each label in y, it sets the corresponding position in one_hot to 1.
    #NumPy pairs each row index with its corresponding column index and assigns all those positions at once using vectorized (advanced) indexing.
    return one_hot

#y_train -> (60000,) -> 60000 labels (0-9)
Y_train = one_hot(y_train)

#Now
#X_train -> (60000, 784)
#Y_train -> (60000, 10)

#Initializing weights
np.random.seed(42) #Fix initialization for reproducibility -> same random numbers each time we run the code

input_size = 784 #Starting neurons
hidden_size = 64 #Hidden layer
output_size = 10 #Output 0-9

#Weights are initialized with small random values to break symmetry
def init_params():
    #w[i][j] -> the weight from the i-th neuron in the previous layer to the j-th neuron in the current layer.
    w1 = np.random.randn(input_size, hidden_size) * 0.01 #First layer weights -> 784 input features (pixels) x 64 hidden neurons, multiplied by 0.01 to keep values small for better training stability
    b1 = np.zeros((1, hidden_size)) #First layer bias -> Can be 0 since it will be updated during training.

    w2 = np.random.randn(hidden_size, output_size) * 0.01 #Second layer weights -> 64 x 10
    b2 = np.zeros((1, output_size)) #Second layer bias
    
    return w1, b1, w2, b2

#Activation functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_z = np.exp(Z - np.max(Z, axis = 1, keepdims=True)) #Subtract Z by max to deal with overflow
    return exp_z / np.sum(exp_z, axis = 1, keepdims=True) #Normalize into probabilities where each row sums to 1, axis=1 means we sum across columns (classes) for each example, keepdims=True keeps the dimensions for broadcasting during division

#Forward propagation
#The forward_prop function computes the activations of the hidden and output layers given the input data and current weights and biases. It returns the pre-activation (z) and post-activation (a) values for both layers, which are needed for backpropagation to compute gradients.
def forward_prop(w1, b1, w2, b2, X):
    z1 = X @ w1 + b1 #@ to multiply matrices -> (60000, 784) x (784, 64) + (1, 64) -> (60000, 64)
    a1 = ReLU(z1) #Apply ReLU activation to hidden layer

    z2 = a1 @ w2 + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2

def categorical_cross_entropy(Y, A2):
    #Add small value (1e-7) to A2 to prevent log(0) which would cause numerical instability. This is a common technique to ensure that the logarithm function does not encounter zero values, which would lead to undefined behavior and NaN (Not a Number) results during training.
    #First, get log of predicted probabilities A2
    #Then multiply by the true labels Y (which are one-hot encoded) so only the correct class contributes to the loss
    #Sum across classes (axis=1) to get the loss for each example
    #Finally, take the mean across all examples to get the average loss.
    return -np.mean(np.sum(Y * np.log(A2 + 1e-7), axis = 1))

def back_prop(z1, a1, z2, a2, w2, Y, X):
    m = X.shape[0] #Get number of training examples to calc average gradient

    dZ2 = a2 - Y #Gradient of loss with respect to z2
    dW2 = (a1.T @ dZ2) / m #Gradient of loss with respect to w2, divided by m to get average gradient
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m #Gradient of loss with respect to b2, sum across examples and divide by m for average
    dZ1 = (dZ2 @ w2.T) * (z1 > 0) #Gradient of loss with respect to z1
    dW1 = (X.T @ dZ1) / m #Gradient of loss with respect to w1, divided by m to get average gradient
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m #Gradient of loss with respect to b1, sum across examples and divide by m for average
    #Full dW1 = X.T @ [(a2 - Y) @ w2.T * (z1 > 0) ] / m

    return dW1, db1, dW2, db2

def update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, learning_rate):
    w1 -= learning_rate * dW1 #Update w1 by subtracting the learning rate multiplied by the gradient dW1. Because gradient points to steepest ascent, we subtract it to move in the direction of steepest descent (minimizing loss).
    b1 -= learning_rate * db1
    w2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2

def gradient_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2 = init_params()

    labels = np.argmax(Y, axis=1)

    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = back_prop(z1, a1, z2, a2, w2, Y, X)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        #Print progress
        if i % 10 == 0:
            #loss = categorical_cross_entropy(Y, a2)

            predictions = np.argmax(a2, axis=1)
            accuracy = np.mean(predictions == labels)

            print(f"Iteration {i}")
            #print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print("-" * 30)
    return w1, b1, w2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X) #Keep only A2 which contains the output probabilities for each class
    predictions = np.argmax(A2, axis=1) #For each row, find the index of the maximum value which corresponds to the predicted class (0-9)
    return predictions

# Testing and evaluation
def test_prediction(index, W1, b1, W2, b2):
    x = X_test[index].reshape(1, -1) #Reshape to (1, 784) to match input shape expected by forward_prop

    _, _, _, A2 = forward_prop(W1, b1, W2, b2, x)

    prediction = np.argmax(A2)
    label = y_test[index]

    print("Prediction:", prediction)
    print("Label:", label)

    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.show()

    return prediction, label

# Calculate accuracy
def get_accuracy(predictions, Y):
    #Accuracy = Number of correct predictions / Total number of predictions
    #Compare prediction with Y --> Where they are equal, we get True (1), where they are not equal, we get False (0)
    #Add up to get all correct predictions. 
    # Dividing by the total number of examples (Y.size) gives us the accuracy as a proportion
    return np.sum(predictions == Y) / Y.size

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=200, alpha=0.1)

#Testing---------
def run_ts():
    num_samples = 10
    correct = 0
    np.random.seed(None) #Reset random seed
    indices = np.random.randint(0, len(X_test), num_samples) #Get random index from test set
    for i in indices:
        prediction, label = test_prediction(i, W1, b1, W2, b2)
        correct += int(prediction == label)

    print(f"Accuracy: {correct}/{num_samples}")

run_ts()
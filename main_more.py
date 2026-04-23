import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0 
X_train = X_train.reshape(-1, 784) 

X_test = X_test / 255.0
X_test = X_test.reshape(-1, 784)

def one_hot(y):
    one_hot = np.zeros((y.size, 10)) 
    one_hot[np.arange(y.size), y] = 1
    return one_hot

Y_train = one_hot(y_train)

np.random.seed(42)

input_size = 784
hidden_size = 64
output_size = 10

def init_params():
    #Use He initialization for ReLU activations.
    w1 = np.random.randn(784, 64) * np.sqrt(2 / 784)
    w2 = np.random.randn(64, 64) * np.sqrt(2 / 64)
    w3 = np.random.randn(64, 10) * np.sqrt(2 / 64)
    b1 = np.zeros((1, hidden_size))
    b2 = np.zeros((1, hidden_size))
    b3 = np.zeros((1, output_size))
    
    return w1, b1, w2, b2, w3, b3

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_z = np.exp(Z - np.max(Z, axis = 1, keepdims=True))
    return exp_z / np.sum(exp_z, axis = 1, keepdims=True)

def forward_prop(w1, b1, w2, b2, w3, b3, X):
    z1 = X @ w1 + b1
    a1 = ReLU(z1)

    z2 = a1 @ w2 + b2
    a2 = ReLU(z2)

    z3 = a2 @ w3 + b3
    a3 = softmax(z3)

    return z1, a1, z2, a2, z3, a3

def categorical_cross_entropy(Y, A3):
    return -np.mean(np.sum(Y * np.log(A3 + 1e-7), axis = 1))

def back_prop(z1, a1, z2, a2, z3, a3, w2, w3, Y, X):
    m = X.shape[0]

    dZ3 = a3 - Y
    dW3 = (a2.T @ dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
    dZ2 = (dZ3 @ w3.T) * (z2 > 0)
    dW2 = (a1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = (dZ2 @ w2.T) * (z1 > 0)
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

def update_params(w1, b1, w2, b2, w3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    w1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return w1, b1, w2, b2, w3, b3

def gradient_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2, w3, b3 = init_params()

    labels = np.argmax(Y, axis=1)

    for i in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(z1, a1, z2, a2, z3, a3, w2, w3, Y, X)
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            #loss = categorical_cross_entropy(Y, a3)
            predictions = np.argmax(a3, axis=1)
            accuracy = np.mean(predictions == labels)
            #print(f"Loss: {loss:.4f}")
            print(f"Iteration {i}")
            print(f"Accuracy: {accuracy:.4f}")
            print("-" * 30)
    return w1, b1, w2, b2, w3, b3

def make_predictions(X, w1, b1, w2, b2, w3, b3):
    _, _, _, _, _, A3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
    predictions = np.argmax(A3, axis=1)
    return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    x = X_test[index].reshape(1, -1)

    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, x)

    prediction = np.argmax(A3)
    label = y_test[index]

    print("Prediction:", prediction)
    print("Label:", label)

    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.show()

    return prediction, label

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, iterations=200, alpha=0.1)

def run_ts():
    num_samples = 10
    correct = 0
    np.random.seed(None)
    indices = np.random.randint(0, len(X_test), num_samples)
    for i in indices:
        prediction, label = test_prediction(i, W1, b1, W2, b2, W3, b3)
        correct += int(prediction == label)

    print(f"Accuracy: {correct}/{num_samples}")

run_ts()
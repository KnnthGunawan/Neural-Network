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

layers = [784, 64, 64, 64, 128, 128, 10] #Can be changed to any architecture (except input and output layers)

def init_params(layers):
    params = {}
    for i in range(1, len(layers)):
        fan_in = layers[i-1]
        fan_out = layers[i]
        params[f'w{i}'] = np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
        params[f'b{i}'] = np.zeros((1, fan_out))
    return params

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_z = np.exp(Z - np.max(Z, axis = 1, keepdims=True))
    return exp_z / np.sum(exp_z, axis = 1, keepdims=True)

def forward_prop(params, X):
    L = len(params) // 2 #Number of layers
    cache = {}
    A = X
    cache['A0'] = A

    for l in range(1, L + 1):
        Z = A @ params[f'w{l}'] + params[f'b{l}'] #Z = W @ A + b
        if l < L: #Last Layer use SoftMax
            A = ReLU(Z)
        else:
            A = softmax(Z)
        
        cache[f'Z{l}'], cache[f'A{l}'] = Z, A

    return cache

def categorical_cross_entropy(Y, A):
    return -np.mean(np.sum(Y * np.log(A + 1e-7), axis = 1))

def back_prop(cache, params, Y, X):
    m = X.shape[0]
    L = len(params) // 2

    dZ = {}
    dW = {}
    db = {}

    dZ[f'dZ{L}'] = cache[f'A{L}'] - Y
    dW[f'dW{L}'] = (cache[f'A{L - 1}'].T @ dZ[f'dZ{L}']) / m
    db[f'db{L}'] = np.sum(dZ[f'dZ{L}'], axis=0, keepdims=True) / m

    for l in range(L - 1, 0, -1):
        dZ[f'dZ{l}'] = (dZ[f'dZ{l + 1}'] @ params[f'w{l + 1}'].T) * (cache[f'Z{l}'] > 0)
        dW[f'dW{l}'] = (cache[f'A{l - 1}'].T @ dZ[f'dZ{l}']) / m
        db[f'db{l}'] = np.sum(dZ[f'dZ{l}'], axis=0, keepdims=True) / m

    return dW, db

def update_params(params, dW, db, learning_rate):
    for i in range(1, len(params) // 2 + 1):
        params[f'w{i}'] -= learning_rate * dW[f'dW{i}']
        params[f'b{i}'] -= learning_rate * db[f'db{i}']
    return params

def gradient_descent(params, X, Y, iterations, alpha):
    labels = np.argmax(Y, axis=1)
    L = len(params) // 2
    for i in range(iterations):
        cache = forward_prop(params, X)
        dW, db = back_prop(cache, params, Y, X)
        params = update_params(params, dW, db, alpha)
        if i % 10 == 0:
            #loss = categorical_cross_entropy(Y, cache[f'A{L}'])
            predictions = np.argmax(cache[f'A{L}'], axis=1)
            accuracy = np.mean(predictions == labels)
            #print(f"Loss: {loss:.4f}")
            print(f"Iteration {i}")
            print(f"Accuracy: {accuracy:.4f}")
            print("-" * 30)
    return params

def make_predictions(X, params):
    L = len(params) // 2
    cache = forward_prop(params, X)
    predictions = np.argmax(cache[f'A{L}'], axis=1)
    return predictions

def test_prediction(index, params):
    L = len(params) // 2
    x = X_test[index].reshape(1, -1)

    cache = forward_prop(params, x)
    A_final = cache[f'A{L}']

    prediction = np.argmax(A_final, axis=1)[0]
    label = y_test[index]

    print("Prediction:", prediction)
    print("Label:", label)

    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.show()

    return prediction, label

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

params = init_params(layers)
params = gradient_descent(params, X_train, Y_train, iterations=200, alpha=0.05)

def run_ts():
    num_samples = 10
    correct = 0
    np.random.seed(None)
    indices = np.random.randint(0, len(X_test), num_samples)
    for i in indices:
        prediction, label = test_prediction(i, params)
        correct += int(prediction == label)

    print(f"Accuracy: {correct}/{num_samples}")

run_ts()
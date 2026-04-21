# Neural Network from Scratch

A simple neural network built from scratch using only NumPy to recognize handwritten digits from the MNIST dataset.

## Overview

This project implements a 2-layer neural network with:
- **Input layer**: 784 neurons (28x28 pixel images flattened)
- **Hidden layer**: 64 neurons with ReLU activation
- **Output layer**: 10 neurons with softmax activation

## Features

- One-hot encoding for labels
- Sigmoid, ReLU, and softmax activation functions
- Categorical cross-entropy loss
- Gradient descent optimization
- Backpropagation
- MNIST dataset support

## Requirements

```
numpy
pandas
matplotlib
keras
```

## Usage

```bash
python main.py
```

The network will train for 200 iterations and output accuracy metrics every 10 iterations, then run 10 test predictions on random samples from the test set.

## How it Works

1. **Forward Propagation**: Input → Hidden (ReLU) → Output (softmax)
2. **Loss Calculation**: Categorical cross-entropy between predictions and true labels
3. **Backpropagation**: Computes gradients for all weights and biases
4. **Weight Update**: Gradient descent to minimize loss
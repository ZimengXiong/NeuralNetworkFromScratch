# August 26th, 2025
# Zimeng Xiong

import numpy as np

imageDimension = 16
hiddenLayerDimension = 16
outputDimension = 2

inputImage = np.zeros((imageDimension, imageDimension))

# inputLayer = np.zeros((imageDimension, imageDimension))
# hiddenLayer = np.zeros((hiddenLayerDimension, hiddenLayerDimension))
# outputLayer = np.zeros((1,outputDimension))

# Define weights & biases for each layer
# in (rows, columns)

# Biases are defined with column vectors: [[1],[2],[3]]: (3, 1) by convention

# Weights are defined as (layerTo, layerFrom), so that
# each layer in neuron in layerTo has layerFrom connections to each
# neuron in the previous layer

weight1 = np.random.randn(hiddenLayerDimension, imageDimension**2)
bias1 = np.random.randn(hiddenLayerDimension, 1)

weight2 = np.random.randn(outputDimension, hiddenLayerDimension)
bias2 = np.random.randn(outputDimension, 1)

# Flatten image
flattenedImage = inputImage.reshape((256, 1))

# a MxN matrix times a NxP matrix will result in a MxP result. N must match.
# We use Matrix Multiplication so that for each hidden neuron we:
# 1. Take the value of every neuron from the input layer
# 2. Multiply each of those input values by its unique weight connecting it to the hidden neuron
# 3. Sum up those products

# weight = [[n1w1, n1w2, n1w3, n1w4],
#           [n2w1, n2w2, n2w3, n2w4]]

# input = [i1]
#         [i2]
#         [i3]
#         [i4]

# weight*input = [neuron 1 weighted sum]
#                [neuron 2 weighted sum]

# neuron 1 weighted sum = (n1w1*i1) + (n1w2*i2) + (n1w3+*i3) + (n1w4*i4)

hiddenLayer = weight1 @ flattenedImage + bias1

# Activation functions: allows the model to model non-linear relationships, otherwise
# there is just a linear relationship between input and output (because its just multiplying
# by a set factor and adding a bias shift)

# ReLU: A(x) = max(0, a)
# Technically non-linear, piecewise linear
# Computationally simple
# Dying ReLU problem (neurons stop learning)
#   If the neuron's weighted sum is always negative for all pieces (often arrived
#   by large negative bias term) in your training set, then that
#   neuron's output will always be zero
#   and effectively stop learning
# Used for hidden layers of deep networks

# Sigmoid A(x) = 1/(1+e^(-x))
# Output ranges between 0 and 1
# Vanishing gradient problem
#   The derivative is always a small number,
#   when backpropagating through many layers, these small numbers
#   get multiplied together over and over again, by the time
#   you reach the final (first) layer, the values are too small
#   for the first layer to get any feedback and stop learning
# Used for output layers for binary classification

activatedHiddenLayer = np.maximum(0, hiddenLayer)

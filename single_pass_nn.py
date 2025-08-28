# August 26th, 2025
# Zimeng Xiong

import numpy as np
import math
import copy

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

hiddenLayerRaw = weight1 @ flattenedImage + bias1

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

# Use ReLU activation
activatedHiddenLayer = np.maximum(0, hiddenLayerRaw)

# Don't use sigmoid because we have multiple output nodes,
# sigmoid is useful when there is on binary output node

# Softmax takes the output from both neurons and fits them to a
# probability distribution that adds up to 1.
# e.g. [1.1, 4.0] -> [0.05, 0.95]

# Don't use any activation function during the forward pass!
# The loss function will have softmax built in

# Calculate raw output, "logits"
outputLayerRaw = weight2 @ activatedHiddenLayer + bias2

# Cross-Entropy Loss function
# Provides huge penalty to wrong answer with high confidence

# Step one: apply softmax function
# Turns unbounded raw inputs to a bounded probability distribution [0,1]

# For each element, it's probability is e^(z\sub{i}) divided by
# the summation of e^(z\sub{j}), where z\sub{j} for all elements

# Step two: apply cross entropy loss
# The cross entropy loss function exaggerates incorrect results
# with large confidence

# Take the summation of y\sub{i} times the natural log of p\sub{i},
# where y\sub{i} is the true probability for class `i` and
# p\sub{i} is the softmax normalized probability for the output layer for
# the corresponding class i, for each class from i to k

# y\sub{i} represents the "correct" answer, and
# p\sub{i} represents the guessed answer amplified logarithmically
# Thus, the log component of the unexpected answer (assuming a one-hot class probability)
# is ignored, as y\sub{incorrect} = 0
# Then, the log component of the expected answer term with y\sub{correct} is used in the
# calculation of the loss.
# If we denote the probability of the correct term (for a one-hot class, the log parameter attached
# to the y\sub{i}=1)
# This means as p\sub{correct} increases, log(p\sub{correct}) approaches 0 (as the range
# for the log function is defined as [-\inf, 0]), the loss result approaches zero as
# the X-Entropy function is inverted
# Counter, as p\sub{correct} decreases, the logarithmic term approaches -inf rapidly
# increasing loss result to +inf

# This mechanism results in a smaller loss result when the correct answer has the higher probability
# and high loss when the correct answer has lower probability

# In this case it is known as a `one-hot` vector as the vector only contains
# ones and zeros.
# E.G. for class vertical, y\sub{horizontal}=0 and y\sub{vertical}=1


def calculateSoftmax(rawOutputLayer):
    # 1. Softmax
    softmaxedProbability = []
    softmaxSummation = 0

    for zj in rawOutputLayer:
        softmaxSummation += math.exp(zj)

    for zi in rawOutputLayer:
        softmaxedProbability.append(math.exp(zi) / softmaxSummation)

    # or alternatively:
    # softmaxedProbability = np.exp(rawOutputLayer)/np.sum(np.exp(rawOutputLayer))
    return softmaxedProbability


def calculateLoss(softmaxedOutputLayer, trueProbability):
    # 2. X-Entropy Loss
    loss = 0
    for yi, pi in zip(trueProbability, softmaxedOutputLayer):
        loss += yi * math.log(pi)

    loss = -loss
    return loss


# Backpropagation
# Goal: calculate the gradient of the loss function w.r.t. each 
# parameter: W1, b1, W2, b2

# Z/A0 -> W1 + b1 -> Z1 -> A1 -> W2 + b2 -> Z2 -> A2

# Step #1: Gradient Loss w.r.t Z2
# Find correct with dZ2 = A2 - Y, where
# A2 is the softmaxxed output matrix,
# Y is the one-hot true probability vector

# A2:
# [0.63] 
# [0.37]

# Y:
# [1]
# [0]

# Step #2a: Find gradient loss w.r.t W2
# Backprop using gradient w.r.t. dZ2 and A1 to find dW2
# dW2 = Z2 @ A1.T

# dZ2:
# [-0.37] 
# [-0.63]

# A1:
# [0.3] 
# [0.2]
# [0.8] 
# [0.9]
# [0.4] 
# [0.1]

# A1.T:
# [0.3, 0.2, 0.8, 0.9, 0.4, 0.1]

# Step #2b: Find gradient loss w.r.t b2
# Back prop using gradient dZ2, adding it to b2
# db2 = dZ2

# Resultant is a matrix in the same shape as W2, that describes 
# for each neuron in the output layer the changes to each weight
# that connects it to the previous layer based on how influencial/activated
# the neuron in the previous layer is

# Step #3a: Find error signal at output and w.r.t. A1
# dA1 = W2.T @ dZ2
# dA1 has the same shape as A1, represents summation of 
# errors for a neutron between the (in this case, two) neurons in the output layer
# dA1:
# [n1w1*dz2neuronError + n1w2*dz2neuronError] ..... []
# We use the summation because the previous layer DOES NOT CARE about whether there is more
# significance placed on one or the other output layer neuron
# becuase the previous layer only has control of the ONE value
# of the neuron, not the weights

# Step #3b: Reverse ReLU filter
# Multiply the errors from dA1 by the deriative at the point
# of the Z1 value in the ReLU function

# Step #4: repeat step #2 to find db1, dW1 

def findGradients(A2, trueProbability, A1, Z1, W2, A0):
    # Find dW2 and db2
    dZ2 = A2 - trueProbability
    dW2 = dZ2 @ A1.T
    db2 = dZ2

    # Find error for A1
    dA1 = W2.T @ dZ2

    # Find passthrough error gradient, dZ1
    dZ1 = dA1 * (Z1 >= 0)

    # Find dW1 and db1 from dZ1
    dW1 = dZ1 @ A0.T
    db1 = dZ1

    return (dW2, db2, dW1, db1)


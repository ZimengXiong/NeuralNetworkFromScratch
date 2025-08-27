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


# Backpropagation: assigning blame to each neuron

# 1. Output Layer
# Output layer is simple to do, outputLayer-trueProbability

# 2. Output Layer blame -> Activated Hidden Layer
# Assign blame based on weight per neuron
# Transpose weight into (hiddenLayerDimension, outputLayerDimension) so that
# for every neuron in the hidden layer, there are outputLayerDimension weights associated with it
# connecting it to the next layer
# Calculate dot product (weighted sum) between the transposed blame matrix and
# blame for the output layer, of dimension (hiddenLayerDimension, 1)
# I.E. transposedWeights @ outputLayer

# 3. Activated Hidden Layer blame -> Hidden Layer through ReLU Activation
# If the neuron != 0 before activation, then its "responsible" for its output, and blame can be assigned to it
#   through the ReLU layer
# If the neuron = 0 before activation, then it had no effect on the next layer, so its blame is blocked

# 4. Repeat step to calculate blame for weights between input and hidden layer

# Gradients are adjustment instructions for a parameter
# They have the same shape as the weight matrix, because they are applied
# onto the weight matrix
# E.G.
# From Layer Gradient: [-0.3],
#                        [-0.5]
# Blamed Layer:          [5],
#                        [6],
#                        [3]
# Weight Matrix:         [0.3, -0.6, 0.8]
#                        [0.2, 0.9, -0.5]
# Gradient Matrix: (fromLayerGradient @ blamedLayer.T)

# To figure out how to calculate the adjustment instructions for a weight, we look at
# its input signal and the output's blame


# Calculate output layer blame
def calculateOutputBackprop(softmaxedOutputLayer, trueProbability):
    return np.array(softmaxedOutputLayer) - np.array(trueProbability)


# Backprop through one layer
def generateBlame(
    fromLayerBlame,
    toLayer,
):
    transposedLayer = toLayer.T
    blamedWeights = transposedLayer @ fromLayerBlame
    return blamedWeights


# Backprop through ReLU
def generateBlameThroughReLU(nonActivatedLayer, blamedWeights):
    nonActivatedBlame = copy.copy(nonActivatedLayer)
    for blameIndex, activatedNeuron in zip(range(blamedWeights), nonActivatedLayer):
        if activatedNeuron == 0:
            nonActivatedBlame[blameIndex] = 0

    # alternatively:
    # nonActivatedBlame = blamedWeights * (hiddenLayerRaw > 0)

    return nonActivatedBlame


def generateGradient(blame, weights):
    return weights.T @ blame


def getGradients(
    inputLayer,
    nonActivatedHiddenLayer,
    activatedHiddenLayer,
    softmaxedOutputLayer,
    trueProbability,
):
    # Blame for output layer
    gradient_Z2 = calculateOutputBackprop(softmaxedOutputLayer, trueProbability)

    # Gradient for W2 with blame from output layer
    gradient_W2 = generateGradient(gradient_Z2, activatedHiddenLayer)
    gradient_b2 = gradient_Z2

    # We need to pass through the blame back through the ReLU gate to calculate W1 and b1
    # to get an accurate assessment of how W1 and b1 affected the next layer

    gradient_W1 = generateGradient(
        generateBlameThroughReLU(nonActivatedHiddenLayer, gradient_W2), inputLayer
    )
    gradient_b1 = gradient_W2

    return zip(gradient_Z2, gradient_W2, gradient_b2, gradient_W1, gradient_b1)

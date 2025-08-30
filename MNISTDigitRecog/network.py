# August 29th, 2025
# Zimeng Xiong

import numpy as np
from typing import Tuple
import random
import idx2numpy


class neuralNetwork:
    def __init__(self) -> None:
        self.imageDimension = 28
        self.hiddenLayerDimension = 16
        self.outputDimension = 10

        # Define learning rates
        self.learningRate = 0.01

        # Define weights
        self.W1 = np.random.randn(self.hiddenLayerDimension, self.imageDimension**2)
        self.b1 = np.random.randn(self.hiddenLayerDimension, 1)

        self.W2 = np.random.randn(self.outputDimension, self.hiddenLayerDimension)
        self.b2 = np.random.randn(self.outputDimension, 1)

        # Define one-hot vector, i.e. true result
        self.Y = np.zeros((10,1))

        # Define layers
        # Input image flattened
        self.A0 = np.zeros((self.imageDimension**2, 1))

        # Hidden & Activated layer
        self.Z1 = np.zeros((self.hiddenLayerDimension, 1))
        self.A1 = np.zeros((self.hiddenLayerDimension, 1))

        # Output layer
        self.Z2 = np.zeros((self.outputDimension, 1))
        self.A2 = np.zeros((self.outputDimension, 1))

        self.dW1 = np.zeros((self.hiddenLayerDimension, self.imageDimension**2))
        self.db1 = np.zeros((self.hiddenLayerDimension, 1))
        self.dW2 = np.zeros((self.outputDimension, self.hiddenLayerDimension))
        self.db2 = np.zeros((self.outputDimension, 1))

        # Loss
        self.loss = 0

        # Gives a 60000, 28, 28 matrix
        self.images = idx2numpy.convert_from_file('/home/zimengx/Code/NNFromScratch/MNISTDigitRecog/data/train-images.idx3-ubyte')

        # Gives a 60000 matrix
        self.labels = idx2numpy.convert_from_file("/home/zimengx/Code/NNFromScratch/MNISTDigitRecog/data/train-labels.idx1-ubyte") 
    
    def saveWeights(self):
        """Save all network state arrays to files"""
        # Save weights and biases
        self.W1.tofile("W1.dat")
        self.W2.tofile("W2.dat")
        self.b1.tofile("b1.dat")
        self.b2.tofile("b2.dat")
        
        # Save layer states
        self.A0.tofile("A0.dat")
        self.Z1.tofile("Z1.dat")
        self.A1.tofile("A1.dat")
        self.Z2.tofile("Z2.dat")
        self.A2.tofile("A2.dat")
        
        # Save gradients
        self.dW1.tofile("dW1.dat")
        self.db1.tofile("db1.dat")
        self.dW2.tofile("dW2.dat")
        self.db2.tofile("db2.dat")
        
        # Save one-hot vector and loss
        self.Y.tofile("Y.dat")
        np.array([self.loss]).tofile("loss.dat")
    
    def loadWeights(self):
        """Load all network state arrays from files"""
        # Load weights and biases with proper reshaping
        self.W1 = np.fromfile("W1.dat").reshape(self.hiddenLayerDimension, self.imageDimension**2)
        self.W2 = np.fromfile("W2.dat").reshape(self.outputDimension, self.hiddenLayerDimension)
        self.b1 = np.fromfile("b1.dat").reshape(self.hiddenLayerDimension, 1)
        self.b2 = np.fromfile("b2.dat").reshape(self.outputDimension, 1)
        
        # Load layer states
        self.A0 = np.fromfile("A0.dat").reshape(self.imageDimension**2, 1)
        self.Z1 = np.fromfile("Z1.dat").reshape(self.hiddenLayerDimension, 1)
        self.A1 = np.fromfile("A1.dat").reshape(self.hiddenLayerDimension, 1)
        self.Z2 = np.fromfile("Z2.dat").reshape(self.outputDimension, 1)
        self.A2 = np.fromfile("A2.dat").reshape(self.outputDimension, 1)
        
        # Load gradients
        self.dW1 = np.fromfile("dW1.dat").reshape(self.hiddenLayerDimension, self.imageDimension**2)
        self.db1 = np.fromfile("db1.dat").reshape(self.hiddenLayerDimension, 1)
        self.dW2 = np.fromfile("dW2.dat").reshape(self.outputDimension, self.hiddenLayerDimension)
        self.db2 = np.fromfile("db2.dat").reshape(self.outputDimension, 1)
        
        # Load one-hot vector and loss
        self.Y = np.fromfile("Y.dat").reshape(self.outputDimension, 1)
        self.loss = np.fromfile("loss.dat")[0]

    def loadImage(self, image: np.ndarray, label):
        self.Y = np.array([[1] if x==label else [0] for x in range(10)])
        self.A0 = image.reshape(self.imageDimension**2, 1)
        self.A0 = self.A0/255

    def calculateSoftmax(self):
        self.A2 = np.exp(self.Z2) / np.sum(np.exp(self.Z2))

    def calculateLoss(self):
        correctClassIndex = np.argmax(self.Y)
        predictedProb = self.A2[correctClassIndex, 0]
        # clip prob to be away from 0 and 1
        safeProb = np.clip(predictedProb, 1e-9, 1 - 1e-9)
        self.loss = -np.log(safeProb)

    def findGradients(self):
        # Find dW2 and db2
        dZ2 = self.A2 - self.Y
        self.dW2 = dZ2 @ self.A1.T
        self.db2 = dZ2

        # Find error for A1
        dA1 = self.W2.T @ dZ2

        # Find passthrough error gradient, dZ1
        dZ1 = dA1 * (self.Z1 >= 0)

        # Find dW1 and db1 from dZ1
        self.dW1 = dZ1 @ self.A0.T
        self.db1 = dZ1

    def forwardPass(self):
        self.Z1 = self.W1 @ self.A0 + self.b1

        # Use ReLU activation
        self.A1 = np.maximum(0, self.Z1)

        # Generate output layer
        self.Z2 = self.W2 @ self.A1 + self.b2
    
        self.calculateSoftmax()

    def printShapes(self):
        print(f"""
        A0: {self.A0.shape} 
        A1: {self.A1.shape}
        A2: {self.A2.shape}
        W1: {self.W1.shape}
        W2: {self.W2.shape}
        b1: {self.b1.shape}
        b2: {self.b2.shape}
        Z1: {self.Z1.shape}  
        Z2: {self.Z2.shape}
        dW1: {self.dW1.shape}
        dW2: {self.dW2.shape} 
        Y: {self.Y.shape}
        """)

    def updateWeights(self):
        """
        Update weights with newParameter = oldParameter - (learning_rate*gradient)
        """
        self.W1 -= self.dW1 * self.learningRate
        self.b1 -= self.db1 * self.learningRate
        self.W2 -= self.dW2 * self.learningRate
        self.b2 -= self.db2 * self.learningRate

    def onePass(self, image: np.ndarray, label):
        self.loadImage(image, label)
        self.forwardPass()
        # self.printShapes()
        self.findGradients()
        self.updateWeights()
        self.calculateLoss()
    
    def passThroughAllImages(self):
        for image, label in zip(self.images, self.labels):
            self.onePass(image, label)

    def train(self, epochs, verbose=True):
        for i in range(epochs):
            # print(f"Epoch {i}")
            self.passThroughAllImages()
            if i % 2 == 0:
                if verbose:
                    print(self.loss)

    def runModel(self, image: np.ndarray, value) -> np.ndarray: 
        self.loadImage(image, value)
        self.forwardPass()
        return self.A2

def toAscii(matrix):
    chars = " .:-=+*#%@"
    # Normalize values to 0-1
    norm = matrix / 255.0
    # Map to character indices
    indices = (norm * (len(chars) - 1)).astype(int)
    # Build lines
    lines = ["".join(chars[i] for i in row) for row in indices]
    return "\n".join(lines)

network = neuralNetwork()
network.loadWeights()

images = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
labels = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")
index = random.randint(0,10000-1)

output = network.runModel(images[index], labels[index])

print(toAscii(images[index]))

print(f"Output:")

for i in range(10):
    print(f"{i}: {output[i]}")
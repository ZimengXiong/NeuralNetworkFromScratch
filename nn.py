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

weight1 = np.random.randn(hiddenLayerDimension, imageDimension**2)
bias1 = np.random.randn(hiddenLayerDimension, 1)

weight2 = np.random.randn(outputDimension, hiddenLayerDimension)
bias2 = np.random.randn(outputDimension, 1)


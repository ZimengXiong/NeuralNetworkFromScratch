from simpleNNFS import neuralNetwork
from simpleNNFS import LineType

# Instantiates a instance of the network
network = neuralNetwork()

def runImage():
    image, trueProb = network.generateTestImage()
    network.forwardPassWithImage(image, trueProb)

    return (network.classification, network.A2[0,0], network.A2[1,0])

def runBatch(batches):
    correct = 0
    for _ in range(batches):
        classification, horizontalConfidence, verticalConfidence = runImage()
        if horizontalConfidence > verticalConfidence and classification == LineType.HORIZONTAL:
            correct+=1
        elif horizontalConfidence < verticalConfidence and classification == LineType.VERTICAL:
            correct+=1
    print(f"Percentage Correct: {correct/batches*100}%")


# Run a pre-training batch
print("Before Training: ")
runBatch(10000)

# Train the network
network.train(1500, verbose=False)

# Run the post training batch
print("After Training: ")
runBatch(10000)

from simpleNNFS import neuralNetwork
from simpleNNFS import LineType
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


print("Before Training: ")
runBatch(10)

network.train(10000, verbose=False)

print("After Training: ")
runBatch(10)

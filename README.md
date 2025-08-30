# Neural Network from Scratch

A Collection of Neural networks implemented using NumPy and Python for learning

[MNISTDigitRecog/](MNISTDigitRecog/)

[MNISTDigitRecog/network.py](MNISTDigitRecog/network.py)

Builds on top of simpleNN (read below) to read from the [MNIST dataset](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/) of handwritten digits and train the same, 3 layer NN with it, this time with a 784 unit input layer, 16 unit hidden layer, and 10 unit output layer, using the same fundamental ReLU and Softmax activation techniques and gradient descent backprop. Weight loading and saving was added due to the longer training times of 10ish minutes, due to being poorly optimized and single threaded, for 200 epochs, each going over the entire 60k image training set.

To run the pre-trained model with a random image chosen from the test data set:

```zsh
(.venv) zimengx@kubuntu ~/C/N/MNISTDigitRecog (main)> python3 network.py



             .*%+
            .%%%%:
            *%%%%-
           =%%%%%#
           %%%- *%+
          *%%=   %%.
         .%%%    +%*
         *%%+    =%#
         %%+.    .%%-
        +%%-      #%=
        %%*       #%#
       .%%:       #%#
       :%%.      .%%#
       =%#       -%%#
       =%#       *%%#
       +%+      +%%%-
       =%*    :#%%%+
       -%%=-+#%%%%=
        %%%%%%%%%=
        :%%%%%%+:


Output:
0: [0.99999999]
1: [3.9535492e-30]
2: [9.89169807e-12]
3: [1.22256901e-15]
4: [1.50059062e-28]
5: [1.59402379e-09]
6: [2.70290338e-16]
7: [3.80512781e-09]
8: [6.44995663e-32]
9: [1.20772999e-17]
```

To replicate training, replace the section following function and class declarations with the following:

```python
network = neuralNetwork()

network.train(200)
network.saveWeights()
```

[simpleNN/](simpleNN/)

A basic from-scratch neural network implemented in Python using NumPy, with one input layer (256 units for a 16×16 image), one hidden layer (16 units with ReLU activation), and one output layer (2 units with Softmax activation). Training is performed using cross-entropy loss with gradients computed via backpropagation, and parameters are updated using stochastic gradient descent, training on synthetic data consisting of horizontal and vertical line images

[simpleNN/simpleNNFS.py](simpleNN/simpleNNFS.py)

To run training:

```zsh
source .venv/bin/activate.fish # may be different for your shell
python simpleNNFS.py
```

You should see the loss value for every 100 epochs of training, adjust epochs and learning rate accordingly in code.

```zsh
zimengx@kubuntu ~/C/NNFromScratch (main)> python3 simpleNNFS.py
/home/zimengx/Code/NNFromScratch/simpleNNFS.py:285: SyntaxWarning: invalid escape sequence '\s'
  # For each element, it's probability is e^(z\sub{i}) divided by
4.263875606794645e-07
0.25112331237697777
2.433364664397769
0.05460800950718282
0.09632613142296638
0.0023292509038307274
0.00891875474974326
0.10522446794641487
0.19704331864279823
0.027309717655647242
0.037006739733842026
```

[simpleNN/runModel.py](simpleNN/runModel.py)

To run tests:

```bash
(.venv) zimengx@kubuntu ~/C/NNFromScratch (main)> python3 runModel.py
Before Training:
Percentage Correct: 40.68%
After Training:
Percentage Correct: 100.0%
```

The script generates and runs batches of 10000 images into the model and reports the percentage of the correctly identified before and after training with 1500 generated images.

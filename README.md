# Neural Network from Scratch

[simpleNNFS.py](simpleNNFS.py)

A basic from-scratch neural network implemented in Python using NumPy, with one input layer (256 units for a 16×16 image), one hidden layer (16 units with ReLU activation), and one output layer (2 units with Softmax activation). Training is performed using cross-entropy loss with gradients computed via backpropagation, and parameters are updated using stochastic gradient descent capable of training on synthetic data consisting of horizontal and vertical line images, notes included.

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


[runModel.py](runModel.py)

To run tests:

```bash
(.venv) zimengx@kubuntu ~/C/NNFromScratch (main)> python3 runModel.py
Before Training: 
Percentage Correct: 40.68%
After Training: 
Percentage Correct: 100.0%
```

The script generates and runs batches of 10000 images into the model and reports the percentage of the correctly identified before and after training with 1500 generated images.
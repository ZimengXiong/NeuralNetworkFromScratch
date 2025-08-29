# Neural Network from Scratch

[simpleNNFS.py](simpleNNFS.py)

Learning attempt at a basic from scratch neural network with 1 input (256x), 1 hidden (16x), and 1 output (2x) layer using ReLU and Softmax activations with gradient calculations using XEntropy written in Python and NumPY capable of training, notes attached.

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

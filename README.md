# The-Evolution-of-Defection
A simple deep RL project on iterated prisoner's dilemma by [Harris Zhang](https://github.com/HarrrrisZhang).

(WIP)

#

python 3.8+ is recommended for various performance boost.

### network module
The network module (inside the network folder) could be used standalone to construct simple ANNs by calling `<import network>`.

This will import the following class objects:
- NeuralNetwork
- Linear_layer
- Activation_layer
- BatchNorm_layer
- Dropout_layer
- Maxout_layer

See tests/nn_test.ipynb for examples.

### tests
MNIST dataset is too large to upload onto github, please manually place them inside the tests folder if you wish to run the tests.

### TODO
- Different input structure
- Find ways to reduce NNPlayer.learn's variance, if possible
- Learn defection

- Try implementing policy gradient
- Try implementing actor critic
- Build a GPU version for network module
# The-Evolution-of-Defection
A simple deep RL project on iterated prisoner's dilemma by [Harris Zhang](https://github.com/HarrrrisZhang).

(WIP)
The project is divided into 3 stages:

Stage 1: What is DQN and how do they learn? (learn IPD against TitForTat)

Stage 2: Can they learn the optimal strategy (TitForTat)? (play against multiple opponents)

Stage 3: Can ANNs learn from each other? (Adversarial training)

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
- Conv1d_layer (stride and backward dx not fully implemented yet)

See examples/neural_network.ipynb for details.


### Progress

- Learnt to cooperate
- Learnt to backstab on the last turn
- Won small tournament against axl.TitForTat & axl.Alternator


### TODO

- Learn retaliating defection
- Learn provocative defection

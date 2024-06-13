### Fully Connected Layer (Linear Layer) in Neural Networks

A fully connected layer, also known as a dense layer or linear layer, is a fundamental building block in neural networks. It is called "fully connected" because every neuron in the layer is connected to every neuron in the previous layer.

### Key Concepts of Fully Connected Layers

1. **Neurons and Connections**: Each neuron in a fully connected layer receives input from all neurons in the previous layer and has a unique set of weights.
2. **Weights and Biases**: Each connection between neurons has a weight, and each neuron has an additional parameter called bias.
3. **Activation Function**: After the weighted sum of inputs is computed, an activation function is often applied to introduce non-linearity.
4. **Forward Pass**: The process of computing the output of a fully connected layer from the input.
5. **Backward Pass**: The process of updating weights and biases using gradients during training.

### 1. Neurons and Connections

In a fully connected layer, each neuron is connected to all neurons in the previous layer. The input to each neuron is the weighted sum of outputs from all neurons in the previous layer plus a bias term.

### 2. Weights and Biases

- **Weights (`W`)**: These are learnable parameters that determine the strength of the connection between neurons. If the input layer has `n` neurons and the fully connected layer has `m` neurons, the weight matrix will be of size `m x n`.
- **Biases (`b`)**: Each neuron in the fully connected layer has an associated bias. The bias is also a learnable parameter that allows the model to fit the data better.

### 3. Activation Function

The activation function introduces non-linearity into the model, enabling it to learn complex patterns. Common activation functions include:
- **ReLU (Rectified Linear Unit)**: `ReLU(x) = max(0, x)`
- **Sigmoid**: `Sigmoid(x) = 1 / (1 + exp(-x))`
- **Tanh**: `Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

### 4. Forward Pass

The forward pass involves calculating the output of the fully connected layer. The output `y` is computed as:

\[ y = W \cdot x + b \]

where:
- `W` is the weight matrix.
- `x` is the input vector.
- `b` is the bias vector.

### Example Calculation

Assume we have:
- 3 input neurons (`x1`, `x2`, `x3`).
- 2 neurons in the fully connected layer (`y1`, `y2`).

Let:
- Weights for `y1` be `W1 = [w11, w12, w13]`
- Weights for `y2` be `W2 = [w21, w22, w23]`
- Biases be `b1` for `y1` and `b2` for `y2`.

The output of the neurons would be:

\[ y1 = w11 \cdot x1 + w12 \cdot x2 + w13 \cdot x3 + b1 \]
\[ y2 = w21 \cdot x1 + w22 \cdot x2 + w23 \cdot x3 + b2 \]

### 5. Backward Pass

During the backward pass, the model updates the weights and biases using the gradients of the loss function with respect to each parameter. This is achieved through the process called backpropagation.

### Example in PyTorch

Here’s how you define and use a fully connected layer in PyTorch:

```python
import torch
import torch.nn as nn

# Define a fully connected layer
fc = nn.Linear(in_features=3, out_features=2)

# Create dummy input
x = torch.tensor([[1.0, 2.0, 3.0]])

# Forward pass
output = fc(x)
print(output)
```

### Explanation

- **nn.Linear(in_features=3, out_features=2)**: This creates a fully connected layer with 3 input features and 2 output features.
- **x = torch.tensor([[1.0, 2.0, 3.0]])**: This creates a dummy input tensor with 1 sample and 3 features.
- **output = fc(x)**: This performs the forward pass, calculating the output of the fully connected layer.

### Summary

A fully connected layer is a core component in neural networks, responsible for connecting each neuron in one layer to every neuron in the next layer. It involves weights, biases, and often an activation function. Understanding the forward and backward passes is crucial for training neural networks. In PyTorch, fully connected layers are implemented using `nn.Linear`.

By grasping these concepts, you'll have a solid foundation for building and understanding more complex neural network architectures.

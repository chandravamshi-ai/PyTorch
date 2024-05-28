

### PyTorch Tensors: A Beginner's Guide

**Introduction

Welcome to the beginner's guide on PyTorch tensors! Tensors are the fundamental building blocks in PyTorch, and understanding them is crucial for any deep learning project. This guide will take you through the basics of tensors, including their creation, manipulation, and various operations. Let's dive in!

**Table of Contents**

1. [What is a Tensor?](#what-is-a-tensor)
2. [Creating Tensors](#creating-tensors)
3. [Basic Tensor Operations](#basic-tensor-operations)
4. [Tensor Manipulation](#tensor-manipulation)
5. [Autograd and Gradients](#autograd-and-gradients)
6. [Moving Tensors to GPU](#moving-tensors-to-gpu)
7. [Saving and Loading Tensors](#saving-and-loading-tensors)
8. [Examples](#examples)

### What is a Tensor?

A tensor is a multi-dimensional array, similar to NumPy arrays, but with additional capabilities for automatic differentiation and GPU acceleration. Tensors are used to encode the inputs and outputs of a model, as well as the model’s parameters.

### Creating Tensors

**From Lists or Arrays**

You can create tensors directly from Python lists or NumPy arrays.

```python
import torch
import numpy as np

# From a list
data = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data)

# From a NumPy array
np_array = np.array(data)
tensor_from_array = torch.tensor(np_array)
```

**Using Built-in Functions**

PyTorch provides several functions to create tensors with specific values.

```python
# Create a tensor filled with zeros
zeros_tensor = torch.zeros((2, 2))

# Create a tensor filled with ones
ones_tensor = torch.ones((2, 2))

# Create a tensor with random values
random_tensor = torch.rand((2, 2))

# Create a tensor with specific values
specific_tensor = torch.tensor([[1, 2], [3, 4]])
```

**Tensor Data Types**

Tensors can have different data types. You can specify the data type when creating a tensor.

```python
# Float tensor
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

# Integer tensor
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
```

### Basic Tensor Operations

**Arithmetic Operations**

Tensors support standard arithmetic operations like addition, subtraction, multiplication, and division.

```python
# Create tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
add_result = a + b

# Subtraction
sub_result = a - b

# Multiplication
mul_result = a * b

# Division
div_result = a / b
```

**Matrix Operations**

Tensors also support matrix operations like matrix multiplication and transpose.

```python
# Create tensors
matrix_a = torch.tensor([[1, 2], [3, 4]])
matrix_b = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
matmul_result = torch.matmul(matrix_a, matrix_b)

# Transpose
transpose_result = torch.transpose(matrix_a, 0, 1)
```

**Tensor Manipulation**

**Reshaping Tensors**

You can change the shape of a tensor using `view` or `reshape`.

```python
# Create a tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Reshape using view
reshaped_tensor_view = tensor.view(3, 2)

# Reshape using reshape
reshaped_tensor_reshape = tensor.reshape(3, 2)
```

**Indexing and Slicing**

You can access specific elements, rows, columns, or sub-tensors using indexing and slicing.

```python
# Create a tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Indexing
element = tensor[0, 1]  # Output: 2

# Slicing
row = tensor[1, :]  # Output: tensor([4, 5, 6])
column = tensor[:, 1]  # Output: tensor([2, 5])
sub_tensor = tensor[:, 1:3]  # Output: tensor([[2, 3], [5, 6]])
```

**Concatenation**

You can concatenate tensors along a specified dimension using `torch.cat` or `torch.stack`.

```python
# Create tensors
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])

# Concatenate along rows (dim=0)
concat_tensor_0 = torch.cat((tensor_a, tensor_b), dim=0)

# Concatenate along columns (dim=1)
concat_tensor_1 = torch.cat((tensor_a, tensor_b), dim=1)

# Stack tensors
stack_tensor = torch.stack((tensor_a, tensor_b), dim=0)
```

### Autograd and Gradients

**Autograd Basics**

PyTorch’s `autograd` package provides automatic differentiation for all operations on tensors. When you set `requires_grad=True`, PyTorch will track all operations on the tensor.

```python
# Create a tensor with gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Perform some operations
y = x + 2
z = y * y * 2

# Compute the gradient
z.backward(torch.tensor([1.0, 1.0, 1.0]))  # Backpropagate
gradients = x.grad  # Output: tensor([ 8., 16., 24.])
```

### Moving Tensors to GPU

You can move tensors to GPU for faster computation using the `.to()` method.

```python
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a tensor
tensor = torch.tensor([1.0, 2.0, 3.0])

# Move tensor to GPU
tensor_gpu = tensor.to(device)
```

### Saving and Loading Tensors

**Saving Tensors**

You can save tensors to disk using `torch.save`.

```python
# Create a tensor
tensor = torch.tensor([1.0, 2.0, 3.0])

# Save tensor to file
torch.save(tensor, 'tensor.pth')
```

**Loading Tensors**

You can load tensors from disk using `torch.load`.

```python
# Load tensor from file
loaded_tensor = torch.load('tensor.pth')
```

### Examples

**Example 1: Basic Tensor Operations**

```python
import torch

# Create tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Perform operations
add_result = a + b
sub_result = a - b
mul_result = a * b
div_result = a / b

print("Addition:", add_result)
print("Subtraction:", sub_result)
print("Multiplication:", mul_result)
print("Division:", div_result)
```

**Example 2: Matrix Multiplication**

```python
import torch

# Create tensors
matrix_a = torch.tensor([[1, 2], [3, 4]])
matrix_b = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
matmul_result = torch.matmul(matrix_a, matrix_b)

print("Matrix Multiplication Result:\n", matmul_result)
```

**Example 3: Reshaping and Concatenation**

```python
import torch

# Create a tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Reshape tensor
reshaped_tensor = tensor.view(3, 2)

# Concatenate tensors
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])
concat_tensor = torch.cat((tensor_a, tensor_b), dim=0)

print("Reshaped Tensor:\n", reshaped_tensor)
print("Concatenated Tensor:\n", concat_tensor)
```

**Example 4: Using Autograd**

```python
import torch

# Create a tensor with gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Perform operations
y = x + 2
z = y * y * 2

# Backpropagate
z.backward(torch.tensor([1.0, 1.0, 1.0]))
gradients = x.grad

print("Gradients:\n", gradients)
```

Example 5: Moving Tensor to GPU

```python
import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a tensor
tensor = torch.tensor([1.0, 2.0, 3.0])

# Move tensor to GPU
tensor_gpu = tensor.to(device)

print("Tensor on GPU:\n", tensor_gpu)
```

**Conclusion**

Tensors are the backbone of PyTorch, enabling a wide range of computations and facilitating the development of deep learning models. 

Let's focus on the concept of weights (W) in a fully connected layer.

### Weights (W) in a Fully Connected Layer

**Definition**: Weights are learnable parameters that determine the strength of the connection between neurons in different layers of a neural network.

### Understanding the Weight Matrix

When you have an input layer with `n` neurons and a fully connected (dense) layer with `m` neurons, the weight matrix will have dimensions `m x n`. This means each neuron in the fully connected layer has a weight for each input neuron.

### Example

Let’s consider a simple example to understand this:

- **Input Layer**: 3 neurons (`x1`, `x2`, `x3`)
- **Fully Connected Layer**: 2 neurons (`y1`, `y2`)

The weight matrix `W` will be of size `2 x 3`.

#### Step-by-Step Calculation

1. **Weight Matrix Dimensions**:
   - Since the input layer has 3 neurons and the fully connected layer has 2 neurons, the weight matrix will be `2 x 3`.

2. **Weight Assignments**:
   - Each neuron in the fully connected layer will have 3 weights, one for each input neuron.

   Let's denote the weights as follows:
   - Weights for neuron `y1`: `w11`, `w12`, `w13`
   - Weights for neuron `y2`: `w21`, `w22`, `w23`

   So, the weight matrix `W` is:
   ```
   W = [ [w11, w12, w13],
         [w21, w22, w23] ]
   ```

3. **Input Vector**:
   - The input vector `x` with 3 neurons is: `x = [x1, x2, x3]`

4. **Calculating Output for Each Neuron**:
   - **Output for neuron `y1`**:
     \[
     y1 = w11 \cdot x1 + w12 \cdot x2 + w13 \cdot x3
     \]

   - **Output for neuron `y2`**:
     \[
     y2 = w21 \cdot x1 + w22 \cdot x2 + w23 \cdot x3
     \]

### Example with Numbers

Let’s use specific numbers to make it clearer:

- Input neurons: `x = [1.0, 2.0, 3.0]`
- Weight matrix:
  ```
  W = [ [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6] ]
  ```

**Calculating the outputs**:

- **For `y1`**:
  \[
  y1 = (0.1 \cdot 1.0) + (0.2 \cdot 2.0) + (0.3 \cdot 3.0)
     = 0.1 + 0.4 + 0.9
     = 1.4
  \]

- **For `y2`**:
  \[
  y2 = (0.4 \cdot 1.0) + (0.5 \cdot 2.0) + (0.6 \cdot 3.0)
     = 0.4 + 1.0 + 1.8
     = 3.2
  \]

So, the outputs of the fully connected layer are `[1.4, 3.2]`.

### Summary

- The weight matrix `W` in a fully connected layer determines the strength of the connections between input neurons and output neurons.
- For an input layer with `n` neurons and a fully connected layer with `m` neurons, the weight matrix will have dimensions `m x n`.
- Each element in the weight matrix represents the connection strength between a specific input neuron and a specific output neuron.
- The output of each neuron in the fully connected layer is computed as the weighted sum of the inputs, where the weights are the corresponding elements in the weight matrix.

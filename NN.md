[Return to contents](https://github.com/devychen/Notes-SNLP/tree/main#readme)

# Neural Networks

## Basics

### Perceptron
_A very good article on [What The Hell Is Perceptron](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53)_
- A linear classifier, has a binary output $0$ or $1$, does NOT have a non-linear activation function (unit-step is a piecewise linear. function).
- Compute outputs as: 
```{math}
 y = 
  \begin{cases}
    0  & \quad \text{if } w\times x + b\leq 0\\
    1  & \quad \text{if } w\times x + b > 0
  \end{cases}
```


#### Logical XOR Problem
- AND, OR, XOR.
  - AND: true if all true
  - OR: true if one true
  - **XOR** ("exclusive or"): true iff one true.
- Not a linearly separable function: Impossible to draw a single line (called **decision boundary**) to seperate the inputs into different categoties, and impossible to build a perceptron to compute XOR. 
  - Rather, we need a layered network of perceptron units.

### Neural Networks (NNs)
- Origins lie in biological inspirations from McCulloch-Pitts neuron.
- VS. Logistic Regression:
  - Has very similar maths to LR. And also a <ins>discriminative classifier</ins>.
  - Different in that it <ins>avoids</ins> most uses of rich hand-derived features 
  as in LR, <ins>instead</ins> building NNs that take raw words as inputs 
  and learn to induce features as part of the process of learning to classify.
- <ins>The use of modern NNs</ins> is often called **deep learning**, "deep" as <ins>have many layers</ins>.
- **Units**: Modern NNs is a network of small computing units, each of which takes a vector of input values and produces a single output value.
- Three popular non-linear function $f()$:
  - **Sigmoid**
  - **Tanh** (Tangible H)
  - **ReLU** (Recified Linear Unit)


## Feed Forward NNs
- An architecture of NN. 
  - While perceptron is purely linear, modern multilayer networks are made up of units with <ins>non-linearities</ins> like sigmoids.
- Definition: a multilayer network in which the units are connected with <ins>no cycles</ins>; the outputs from units in each layer are passed to units in the next higher layer, and <ins>no outputs are passed back to lower layers</ins>. (In contrast to RNN which with cycles.)
#### The Architecture
- Three kinds of nodes/layers: input (units), hidden ~, output ~.
- Each layer is **fully connected**:
  - Each unit in each layer takes as input the outputs from the previous layer;
  - There's <ins>a linke between every pairs of units</ins> from two adjacent layers.
- Hidden layer (the core):
  - **Hidden unit** each has a weight vector and a bias as parameters, each sums over all the input units. 
  - Hidden layer thus form a representation of the input.
  - The vector $h$ (the output of hidden layer):
  $h=\sigma(Wx+b)$ (if using sigmoid as af)
- Notation for dimentionality:
  - Input layer - layer 0 <br> hidden layer - layer 1 <br> output layer - layer 2
  - $x\in \mathbb{R^{n_0}}$: $x$ is a vector of real numbers of dimention $n_0$
  - a column vector of dimensionality $[n_0, 1]$
  

## Training NNs


[Return to contents](https://github.com/devychen/Notes-SNLP/tree/main#readme)

# Neural Networks

## Basics

### Perceptron
_A very good article on [What The Hell Is Perceptron](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53)_
- A linear classifier, has a binary output $0$ or $1$, does NOT have a non-linear activation function (unit-step is a piecewise linear. function).
- Compute outputs as: 
```math
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
- Definition: a multilayer network in which the units are connected with <ins>no cycles</ins>; the outputs from units in each layer are passed to units in the next higher layer, and <ins>no outputs are passed back to lower layers</ins>. (In contrast to RNN which with cycles)
#### The Architecture
- Three kinds of nodes/layers: input (units), hidden ~, output ~.
- Each layer is **fully connected**:
  - Each unit in each layer takes as input the outputs from the previous layer;
  - There's <ins>a link between every pair of units</ins> from two adjacent layers.
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


## Additional Materials:
[Interactive 3Blue1Brown Series on DL](https://www.3blue1brown.com/topics/neural-networks)

#### Introduction
  > Somehow identifying digits is incredibly easy for your brain to do, but almost impossible to describe how to do. The traditional methods of computer programming, with if statements and for loops and classes and objects and functions, just don’t seem suitable to tackle this problem. But what if we could write a program that mimics the structure of your brain? That’s the idea behind neural networks. <br>

  > Moreover, just as you learn by seeing many examples, the “learning” part of machine learning comes from the fact that we never give the program any specific instructions for how to identify digits. Instead, we’ll show it many examples of hand-drawn digits together with labels for what they should be, and leave it up to the computer to adapt the network based on each new example.<br>

#### Structure
*(Sources: [What is NN? (3Blue1Brown)](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1) and a recommended [online book](http://neuralnetworksanddeeplearning.com/) on NN and DL.)*

- Many <ins>variants</ins>: convolutional NNs, recurrent NNs, transformers, ...
  - Plain vanilla form, aka MLP (multilayer perceptron)
- **Neuron**: a thing that holds a number, between $0.0 - 1.0$. 
  - NN are just a bunch of neurons connected together. The higher the lighter the active.
  - This number inside is called "activation" of that neuron, representing the inputs & outputs of network, it has influence on the activation of each neuron in the next layer. <span style="color:lightgray">(e.g. a pixel)</span>
- Why using layers
  - Determine how strong the connectons between layers are at the heart of how NN operates. 

#### Learning (how to train it with labeled examples)<br>
*(Source: [What is Gradient Descent?](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3))*

#### Backpropagation
*(Source: [What is bp doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)*
[Return to contents](https://github.com/devychen/Notes-SNLP/tree/main#readme)

# Neural Networks

## Basics

### Perceptron
<span style="color:lightgray">_A very good article on [What The Hell Is Perceptron](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53)_</span>
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


## Additional Materials
Below are notes on the Interactive 3Blue1Brown Series on DL, [written version](https://www.3blue1brown.com/topics/neural-networks), and the [video](https://www.3blue1brown.com/topics/neural-networks)

#### Introduction
  > Somehow identifying digits is incredibly easy for your brain to do, but almost impossible to describe how to do. The traditional methods of computer programming, with if statements and for loops and classes and objects and functions, just don’t seem suitable to tackle this problem. But what if we could write a program that mimics the structure of your brain? That’s <ins>the idea behind neural networks</ins>. <br>

  > Moreover, just as you learn by seeing many examples, the “learning” part of machine learning comes from the fact that we never give the program any specific instructions for how to identify digits. Instead, we’ll <ins>show it many examples of hand-drawn digits together with labels for what they should be, and leave it up to the computer to adapt the network based on each new example</ins>.

#### Structure
*(Sources: [What is NN? (3Blue1Brown)](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1) and a recommended [online book](http://neuralnetworksanddeeplearning.com/) on NN and DL.)*

- Many <ins>variants</ins>: C<span style="color:gray">onvolutional</span>NNs, R<span style="color:gray">ecurrent</span>NNs, transformers, ...
  - Plain vanilla form, aka **MLP (multilayer perceptron)**
- **Neuron**: a thing that holds a number, between $0.0 - 1.0$. 
  - NN are just a bunch of neurons connected together. The higher the lighter the active.
  - This number inside is called "activation" of that neuron, representing the inputs & outputs of network, it has influence on the activation of each neuron in the next layer. <span style="color:lightgray">(e.g. a pixel)</span>
- Why using layers
  - Determine how strong the connectons between layers are at the heart of how NN operates. 
  - <ins>The motivation</ins>: We hope that the layered structure might <ins>allow the problem to be broken into smaller steps</ins>, from pixels to edges to loops and lines, and finally to digits:
    - <span style="color:gray">i.e. The second layer could pick up on edges, the third on patterns like loops and lines, and the last one pieces together those patterns to recognize digits</span>.
- How information passes between layers (**weights**)
  - > what parameters the network should have, what knobs and dials you should be able to tweak, so that it’s expressive enough to potentially capture the pattern.
  - Assign **weights** (just numbers) to each neuron, each weight is an indication of how its neuron in the first layer is <ins>correlated with</ins> this new neuron in the second layer.
  - the hope is that if we add up all the desires from all the weights, the end result will be a neuron that does a reasonably good job of detecting what we’re looking for.
  - So to actually compute the value of this second-layer neuron, we take all the activations from the neurons in the first layer, and compute their weighted sum:
    $w_1a_1+w_2a_2+w_3a_3+...+w_na_n$
  - the result is a number, we need to squish the real number line into the range between $0 - 1$, thus - **Sigmoid Squashification $\sigma$** <span style="color:lightgray">(S型壓縮)</span> or other <ins>**activation** functions (the activation of the neurons)</ins> such as ReLU <span style="color:lightgray">(both non-linear)</span>.
    - Very negative inputs end up close to $0$
    - very positive inputs end up close to $1$
    - It steadily increases around $0$. <br>
    i.e. $\sigma(-1000)$ is close to $0$
    - So the activation of the neuron here will basically be a measure of <ins>how positive the weighted sum is</ins>. 

![Image](/pics/sigmoid.png)

  - But maybe it’s not that we want the neuron to light up when this weighted sum is bigger than 0. Maybe we only want it to be meaningfully active when that sum is bigger than, say, 10. That is, we want some **bias** for it to be inactive.  <br>
  
![Image](/pics/bias.png)
  
  - So:
    - the weights tell you this neuron in the second layer is <ins>picking up on what pattern </ins>, and 
    - the bias tells you <ins>how big that weighted sum needs to be</ins> before the neuron gets <ins>meaningfully active</ins>. <br>
  
![Image](/pics/dimensionality.png)

- **Math representation/Formal annotation**
  > Each neuron is a function, it takes in the activations of all neurons in the previous layer, and spits out a number between $0 - 1$. <br>
  > The entire network is just a function too! It takes in 784 numbers as its input, and spits out 10 numbers as its output. It’s an absurdly complicated function, because it takes over 13,000 parameters (weights and biases), and it involves iterating many matrix-vector products and sigmoid squishificaitons together. But it’s just a function nonetheless. <br>

<img src="/pics/neural-network-function.png" width = "900">

  - Using matrixes as a whole instead of one-by-one: <br>

![Image](/pics/mathrepre1.png) 
![Image](/pics/mathrepre2.png)

#### Learning (how to train it with labeled examples)<br>
*(Source: [What is Gradient Descent?](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3))*
> By learning, we mean getting the computer to find an optimal setting for all these parameters that will solve the problem at hand.
> ML vs CS: no instructions/algorithms on how, but algorithms on taking labeld example to adjust parameters and perform better. 
- Training data: the examples given
- 





#### Backpropagation
*(Source: [What is backpropagation doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4))*
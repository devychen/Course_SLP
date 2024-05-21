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
> Below are notes based on the Interactive 3Blue1Brown Series on DL, [written version](https://www.3blue1brown.com/topics/neural-networks), and the [video](https://www.3blue1brown.com/topics/neural-networks).<br> He's also recommended an [online book](http://neuralnetworksanddeeplearning.com/) on NN and DL.

### Introduction
  > Somehow identifying digits is incredibly easy for your brain to do, but almost impossible to describe how to do. The traditional methods of computer programming, with if statements and for loops and classes and objects and functions, just don’t seem suitable to tackle this problem. But what if we could write a program that mimics the structure of your brain? That’s <ins>the idea behind neural networks</ins>. <br>

  > Moreover, just as you learn by seeing many examples, the “learning” part of machine learning comes from the fact that we never give the program any specific instructions for how to identify digits. Instead, we’ll <ins>show it many examples of hand-drawn digits together with labels for what they should be, and leave it up to the computer to adapt the network based on each new example</ins>.

### Structure
- Many <ins>variants</ins>: C<span style="color:gray">onvolutional</span>NNs, R<span style="color:gray">ecurrent</span>NNs, transformers, ...
  - Plain vanilla form, aka **MLP (multilayer perceptron)**
- **Neuron**: a thing that holds a number, between $0.0 - 1.0$. 
  - NN are just a bunch of neurons connected together. The higher the lighter the active.
  - This number inside is called "activation" of that neuron, representing the inputs & outputs of network, it has influence on the activation of each neuron in the next layer. <span style="color:lightgray">(e.g. a pixel)</span>
- Why using layers
  - Determine how strong the connectons between layers are at the heart of how NN operates. 
  - <ins>The motivation</ins>: We hope that the layered structure might <ins>allow the problem to be broken into smaller steps</ins>, from pixels to edges to loops and lines, and finally to digits:
    - <span style="color:gray">i.e. The second layer could pick up on edges, the third on patterns like loops and lines, and the last one pieces together those patterns to recognize digits</span>.

![Image](/pics/each-layer.png)

- How information passes between layers (**weights**)
  - > what parameters the network should have, what knobs and dials you should be able to tweak, so that it’s expressive enough to potentially capture the pattern.
  - Assign **weights** (just numbers) to each neuron, each weight is an indication of how its neuron in the first layer is <ins>correlated with</ins> this new neuron in the second layer.
  - Relations:
    - Increasing one of those weights of a connection with a bright neuron has a bigger influence on the cost function than increasing the weight of a connection with a dimmer neuron.
    - <ins>The biggest increases in weights</ins>—the biggest strengthening of connections—happens between neurons that are <ins>the most active</ins> and the ones which we wish to <ins>become more active</ins>.
  - The hope is that if we add up all the desires from all the weights, the end result will be a neuron that does a reasonably good job of detecting what we’re looking for.
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
    - Picture below shows the example dimensionality of $w$ and $b$.
  
![Image](/pics/dimensionality.png)

- **Math representation/Formal annotation**
  > Each neuron is a function, it takes in the activations of all neurons in the previous layer, and spits out a number between $0 - 1$. <br>
  > The entire network is just a function too! It takes in 784 numbers as its input, and spits out 10 numbers as its output. It’s an absurdly complicated function, because it takes over 13,000 parameters (weights and biases), and it involves iterating many matrix-vector products and sigmoid squishificaitons together. But it’s just a function nonetheless. <br>

![Image](/pics/neural-network-function.png)

  - Using matrixes as a whole instead of one-by-one: <br>

![Image](/pics/mathrepre1.png) 
![Image](/pics/mathrepre2.png)

### Learning (how to train it with labeled examples)
> By **learning**, we mean getting the computer to find an optimal setting for all these parameters that will solve the problem at hand. <br>
> ML vs CS: no instructions/algorithms on how, but algorithms on taking labeld example to adjust parameters and perform better. <br>
> Recall: our network's behavior is determined by all of its weights and biases. The <ins>weights</ins> represent the strength of connections between each neuron in one layer and each neuron in the next. And each <ins>bias</ins> is an indication of whether its neuron tends to be active or inactive.
- Training data: the examples given
- **Gradient Descent**
  - **Gradient**, a term from calculus. It's a <ins>vector</ins> that tells you which direction you should step to increase the function most quickly.
  - **More frankly, understand GD as a way to compute this vector, which tells you both what the “downhill” direction is, and how steep it is.**
  - By following the slope (moving in the downhill direction), we approach a local minimum.
  - The image to have in mind is a ball rolling down a hill (see below)
  - We start with one random input (could be ≥2), so <ins>there are many possible valleys you might land in</ins> depending on which random input you start at, and there’s no guarantee that the local minimum you land in will be the smallest possible value for the cost function
  - If you <ins>make your step sizes proportional to the slope itself</ins> when the slope flattens out towards a minimum, your steps will get smaller and smaller, and that keeps you from overshooting.
  - In practice, each step will look like $\eta\Delta C$ where the constant $\eta$ is known as the **learning rate**. <ins>The larger it is, the bigger your steps, which means you might approach the minimum faster</ins>, but there’s a risk of overshooting and oscillating a lot around that minimum.
  - Steps: 
    1. Initialisation: start with random values for $w$ & $b$;
    2. Computation gradient: calculate the gradient with respect to each parameter, taking partial derivatives of the cost function with respect to $w$ & $b$;
    3. Update parameters $\theta$: Adjust the parameters (could be a $w$ or $b$) in the direction opposite to the gradient.
    4. Iterate: Repeat the gradient computation and parameter update steps until convergence, i.e., until the changes in the cost function are smaller than a pre-defined threshold, or for a fixed number of iterations.

![Image](/pics/gradient-ball.png)

### Limitations
Classical neural networks has a constrained training environment. Every digit that it saw was of a particular size, centered in the frame. So if you give it something that is too big or too small or not quite in the center, it is bound to get confused.<br>
Our current learning algorithm does nothing to let the network transfer knowledge of patterns picked up on one region of the grid to another, or to make inferences about scaling. In fact, nothing about our training algorithm even uses the fact that some pixels are adjacent to others.

If you start thinking hard about how to change structure of this network to allow for more flexible learning, e.g. how learning a pattern in one part of the image could naturally transfer to any other part of the image, you’ll be well situated to learn about some of the more modern variations on this theme, most notably convolutional neural networks.


### Backpropagation (反向传播)
- **Backpropagation** is an algorithm for computing that negative gradient of the network's cost function.
  - **Negative gradient** of the cost function is a (multi-)dimensional <ins>vector</ins> that tells us <ins>how to push all the weights and biases to decrease the cost most efficiently</ins>.
    -  An interpret example: the component associated with the weight on one neuron comes out to be 3.2, while the component associated with some other neuron is 0.1. <ins>The way to read this is that the cost function is 32 times more sensitive to changes to that first weight</ins>. So if you were to wiggle the value of that weight a bit, it’ll cause a change to the cost function 32 times greater than what the same wiggle to the second weight would cause.
  - BackP helps to find the derivatives - <ins>the entries/elements of the gradient vector</ins> are the partial derivatives of cost function $C$ with respects to every $weights$ and $bias$ in the network. 
- The <ins>intuition</ins> is that, there are 3 avenues together to increase a neuron's activation/cost function:
  1. Increase the bias: 
    - Increase the bias associated with the wanted neuron and decrease those with all the other neurons.
  2. Increase the weights
    - Increasing one of those weights of a connection with the brightest neuron (than increasing the weight of a connection with a dimmer neuron).
  3. Change the activations from the previous layer
    - Increasing activations of all neurons in the previous layer in proportion to the corresponding weights.
    - (Note that we can’t influence the activations in the previous layer directly. But we can change the weights and biases that determine their values.)

![Image](/pics/simple-weights-and-biases.svg)

- Its calculation (chain rule in context of NN in ML)
  - Basically, we want to expore how sensitive the $cost$ is to small changes in the $weight$.
    - $C_0 = (a^{(L)} - y)^2$, if y is the expected output.
    - $a^{(L)} = \sigma (z^{(L)})$
      - $z$ is the weight sum: $z=w^{(L)}a^{(L-1)}+b^{(L)}$
  - $\frac{\partial C_0}{\partial w^{(L)}}$: 
    - The ratio of a tiny change to $z^{(L)}$ to the tiny change in $w^{(L)}$.
    - That is, <ins>the **derivative** of $z^{(L)}$ with respect to $w^{(L)}$</ins>. It is the derivatives that we actually want.
    - The nudge to $w^{(L)}$ has a chain of effects which eventually causes a nudge to $C_0$. 
  - The chain rule ($w \to z \to a \implies C$), where multiplying these three ratios gives us the sensitivity of cost to small changes in weights. See picture below:

  ![Image](/pics/chain-rule-breakdown.svg)

  - The constiuent parts of $\frac{\partial C_0}{\partial w^{(L)}}$ (see picture below for individual formulas):
    - $\frac{\partial C_0}{\partial w^{(L)}} = a^{(L-1)} \sigma' (z^{(L)})2(a^{(L)}-y)$
    - Attention that $\frac{\partial C_0}{\partial a^{(L)}}$ is proportional to the difference between the actual output and the desired output. This means that when the actual output is way different than what we want it to be, even small nudges to the activation stand to make a big difference to the cost.

  ![Image](/pics/each-part-equation.png)
  - The full cost function for the network ($C$) is the average of all the individual costs for each training example:
    - $C = \frac{1}{n} \sum_{\substack{k=0}}^{n-1} C_k$
  - And so the derivative of $C$ is the average of all individual derivatives, it tells us how the overall cost of the network will change when we wiggle the last weight.
    - $\frac{\partial C}{\partial w^{(L)}} = \frac{1}{n} \sum_{\substack{k=0}}^{n-1} \frac{\partial C_k}{\partial w^{(L)}}$ 


# Quick Summary
- **Activation function**: a mathematical operation that transforms the input into an output signal. This output signal is what we call the "**activation**".
  - The **activation value** is defined as a weighted sum of all activations from the previous layer, plus a bias, which is then plugged into activation function.
- **Cost/loss function**: a math function that quantifies the error between predicted and actual values, it is designed to measure network performance. We change the weights and biases to decrease the **cost**.
  - The **cost** for a single training example is the sum of the squares of the differences between the actual output and the desired output.
-  The parameters to be optimized:
    - The **weights** represent the strength of connections between each neuron in one layer and each neuron in the next. It determines the influence of each feature on the output.
    - Each **bias** is an indication of whether its neuron tends to be active or inactive. It allows the model to fit the data even when all feature values are zero.
    - For an extremely simple NN where each layer has just one neuron, the network is determined by 3 $weights$ (one for each connection) and 3 $bias$ (one for each neuron except the 1st). 
- **Gradient Descent**: The method to minimising the cost function by computing the **gradient** with respect to model parameters w&b.
  - **Gradient**: a <ins>vector</ins> of partial derivatives of the cost function with respect to the model parameters. It points in the direction of the steepest ascent of the cost function as so to find the minimal value (of w & b?), namely it points in the direction of increasing cost, so moving in the opposite direction to reduce the cost.
    - More frankly, gradient tells you how <ins>sensitive</ins> the cost function is to each corresponding weight and bias
  - **Learning rate** measures the step, the larger the bigger. 
- **Backpropagation**


<br>
<br>

# Others
### Derivatives

### Partial Derivatives


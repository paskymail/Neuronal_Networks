---
theme : "night"
transition: "slide"
highlightTheme: "monokai"
logoImg: "./images/logo_US.png"
slideNumber: false
title: "Recurrent neural networks for ornithopter trajectory optimization"
---

::: block
*Work in progress* {style=background:red;width:500px}
::: 

---

## Trabajo Fin de Master

 Recurrent neural networks for ornithopter trajectory optimization

 <img data-src="./images/logo_US.png">

<small>Supervisor: José Miguel Díaz Báñez / Candidate: Luis David Pascual Callejo</small>

---

## Overview

1.  Ornithopter trajectory optimization problem
    1.  OSPA approach
    2.  RNN approach
2.  Neural Networks preliminaries
    1.  Feedforward Neural Networks
    2.  NN's parametric family
    3.  Universal Approximator theorem
    4.  Neural Network architecture
    5.  Recurrent Neural Network
--

## Overview

1.  Propositions and basic results
    1.  Unfolding of RNN
    2.  Maximum likelihood method
    3.  ML for the regression problem
    4.  ML for the classification problem
2.  RNN implementation
3.  Results

---

## 1. Ornithopter trajectory optimization problem

<img data-src="./images/ornithopter.jpg" width="40%">

---

### 1.1 Problem statement

<img data-src="./images/trajectories.png" width="40%">

> <small>The problem we want to solve is to compute an optimal trajectory of an ornithopter connecting two given positions A and B while minimizing the energy consumption</small>

---

### 1.2 The OSPA approach 

OSPA (Ornithopter Segmentation Path Planning Approach) is a novel heuristic algorithm able to efficiently compute optimal trajectories. 
However it is too slow to be embarked on the ornithopter for inline path computation.  

---

### 1.3 The RNN approach

> **Proposition:** contour this problem using recurrent neuronal networks.

The neuronal network is tasked with learning the underlying optimal trajectory flight dynamics, which are in turn numerically estimated by the OSPA.


--

### 1.3 The RNN approach

More precisely, OSPA is used to compute a set of optimal trajectories for the ornithopter and then, the neuronal network is tasked with learning the underlying function from it. 

> The goal is to obtain similar performances to the heuristic method with much faster computation times.

---

## 2. Neural Networks preliminaries

---

### 2.1 Feedforward neural networks

<img data-src="./images/NN_schema.png">

--

### 2.1 Feedforward NN algebraic equations

<img data-src="./images/Neuron_diagram.png" width="40%">
<small>
<span>
\[\begin{aligned}
&a_i^k = b_i^k + \sum_{j = 1}^{r_{k-1}} w_{ji}^k o_j^{k-1} = \sum_{j = 0}^{r_{k-1}} w_{ji}^k o_j^{k-1}\\
&o_i^k = g(a_i^k)
\end{aligned} \]
</span>
</small>
---

### 2.2 NN's parametric family

> These equations form a parametric family $\{f^*(\cdot \,;\theta )\mid \theta \in \Theta \}$.

where the family $f^*(\cdot \,;\theta)$ is given by the NN architecture and the parameters $\theta$ correspond to the NN weights $w_{ji}^k$.


--

### 2.2 NN's parametric family

> The goal can be reformulated as getting the best approximate $f^{\ast}(x;\hat{\theta})$ to the true underlying function $f$, which is in turn characterized by the OSPA optimal trajectory data set.


---

### 2.3 Universal approximator theorem

> <small>**Theorem:** feedforward networks with a linear output layer and at least one hidden layer with any continuous squashing function can approximate any Borel measurable function from one finite-dimensional space to another with any desired non-zero amount of error, provided that the network is given enough hidden units.</small>

--

### 2.3 Universal approximator theorem

> Therefore $f^{\ast}(x;\hat{\theta})$ can approximate $f$ as much as desired
 
This is true since a continuous function on a closed and bounded subset of $R^N$ is Borel measurable.

---

### 2.4 NN architecture

The neuronal network architecture and activation functions define the capacity the parametric family 
$$\{f^\ast(\cdot \,;\theta )\mid \theta \in \Theta \}$$ 

<small>
The capacity is determined by:

* Depth and width of the network: they will define the number of parameters available.
* Activation functions and general architecture: they will define the set of functions that can be learned by the NN.
</small>


--

### 2.4 NN architecture

Due to the complexity of the ornithopter problem, our NN architecture must have the capacity to capture temporal dynamic behaviors.

> **Proposition:**  use a recurrent neural network with just one single layer to learn the OSPA underlaying trajectory flight dynamics 

---

### 2.5 Recurrent Neural networks

<small>
A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. These connections allow previous outputs to be used as inputs while having hidden states.
</small>
--

### 2.5 Recurrent Neural networks

<img data-src="./images/RNN gates.png" width="50%">
<small>
<span>
\[\begin{aligned}
&a_t =g_1( b_a + W_{aa}a_{t-1}+W_{ax}x_t) \\
&y_t = g_2(b_y + W_{ya}a_t)
\end{aligned} \]
</span>
</small>
---

## 3. Propositions and basic results

---

## 3.1 RNN unfolding

<img data-src="./images/RNN architecture.png">

---

## 3.1 RNN unfolding
<small>

> **Theorem:** the unfolding property can only be applied if the following hypothesis is met: the conditional probability distribution over the variables at t+1 given the variables at time t, is stationary.
</small>
--

### 3.1 RNN unfolding

> <small>Therefore the RNN can be virtually considered as a feedforward NN and the results seen in [1](#/1) and [2](#/3) are still valid.</small>
 
<small>This is true since for every tuple of starting and target points, it is expected to obtain the same optimal trajectory.  Therefore, considering the starting point as an intermediate point at time t of a longer trajectory with same target point,the following intermediary points are expected to be same.</small>

---

## 3.2 Maximum likelihood

<img data-src="./images/ML concept.png">

---


### 3.2 ML to estimate optimal parameters
<small>
Now that our family is defined, we need to compute the parameters $\theta$ so to obtain the best approximate $f^{\ast}(x;\hat{\theta})$ to the true underlying function $f$.


> **Proposition:** the ML method is proposed to estimate the parameter value $\hat{\theta}$ for a given family, so that under the assumed model $f^{\ast}(x;\hat{\theta})$, the observed data is the most probable. 
</small>
--

### 3.2 ML to estimate optimal parameters

> <small> **Definition:** Consider a set of $m$ examples $X ={x^1,...,x^m} $ drawn independently from the true but unknown data generating distribution $p_{data}(x)$. Let  $p_{model}(X; \theta)$ be a parametric family of probability distributions over the same space indexed by $\theta$. The maximum likelihood estimator for $\theta$ is then deﬁned as: 

<script type="math/tex; mode=display">
  \theta_{ML} =  \arg\max_{\theta}  p_{model}(X;\theta) = \arg\max_{\theta}
\prod_{i=1}^m p_{model}(x_i ;\theta)
</script>

</small>

---

### 3.3 Regression problem

<small>
In the regression problem, the RNN aims to output the best possible approximation to the values of the true states in $R^6$ given by an OSPA trajectory.

$$y_i = \hat{y_i} + e_i$$

where  $\hat{y_i}$ is our prediction, $y_i$ is the real value and $e_i$ is the error due to non modeled aspects which is assumed to be Gaussian
</small>

--

## 3.3.1 Mean squared Error

> <small> **Proposition:** given the above mentioned hypothesis, it is equivalent to use the Log Likelihood or the Mean Squared Error as loss functions for our NN.

<script type="math/tex; mode=display">
   \theta_{ML} =  \arg\max_{\theta} L= \arg\max_{\theta}m\log\frac {1}{\sigma {\sqrt {2\pi }}}-\sum_{i=1}^m {\frac {1}{2}}\left({\frac {y_i - \hat{y_i}  }{\sigma }}\right)^{2} = \arg\max_{\theta}\sum_{i=1}^m {\frac {1}{2}}\left({y_i - \hat{y_i}  }\right)^{2}
</script>

</small>

--

### 3.3.2 KL Divergence

> <small> **Proposition:**  It is equivalent to use the Log-likelihood or the KL divergence as loss functions to compute the optimal parameters for our NN regression problem. 

<script type="math/tex; mode=display">
   \displaystyle D_{\text{KL}}(p_{data}\parallel p_{model})=-H_{p_{data}} - L(x;\theta)
</script>

</small>

---

### 3.3.2 KL Divergence

<small>
The KL divergence loss function represents the amount of information lost when $p_{model}$ is used to approximate $p_{data}$.

Or in other words, when our neural network $f^{\ast}(x;\hat{\theta})$ is used instead of the real source of data, which is our OSPA planner.
</small>
---

### 3.4 Classification problem
<small>
The ornithopter possible action data set is actually a finite set of 35 different action tuples. This is due to the methodology of the OSPA to compute the optimal path, which requires a finite set of possible action outcomes in order to obtain a search tree.

In the classification model each action $a_k$ is treated as a different category, leading to 35 different categories: $$a_k = (\delta_k, f_k), \ k =1,..., 35$$

</small>
--

### 3.3 Classification problem

<small>
In this case, we want the RNN to output the probability that each category has to be selected:
 $$\hat{y_k} = p(y_k|x,\theta)$$
where $\hat{y_k}$ is the vector with the predicted probabilities for each category $k$ and $y_k$ is the "one hot" representation of each category
</small>

--

### 3.4.1 Cross Entropy

> <small> **Proposition:** It is equivalent to use the Log-likelihood or the CategoricalCross Entropy as loss function for our NN classification problem. 

<script type="math/tex; mode=display">
   \theta_{ML} =  \arg\max_{\theta} \sum_{i=1}^m \log p(y|x_i ;\theta)= \arg\max_{\theta} - \sum_{i=1}^m H(p(y_i), \hat{y}_i)
</script>

</small>

--

### 3.4.1 Cross Entropy

<small>
The cross entropy loss function can be interpreted as the expected message-length per datum when a wrong distribution $p_{model}$ is assumed while the data actually follows a distribution $p_{data}$.

Or in other words, when our neural network $f^{\ast}(x;\hat{\theta})$ is used instead of the real source of data, which is our OSPA planner.
</small>

--

### 3.4.2 KL Divergence

> <small> **Proposition:**  It is equivalent to use the Log-likelihood or the KL divergence as loss functions to compute the optimal parameters for our NN classification problem. 

<script type="math/tex; mode=display">
   \displaystyle D_{\text{KL}}(p(y)\parallel \hat{y})=-H_{y} + H_{y\hat{y}}(x;\theta)
</script>

</small>

---

# Questions?
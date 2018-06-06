# Cost Function

Let's first define a few variables that we will need to use:

* L = total number of layers in the network
* sl = number of units (not counting bias unit) in layer _l_
* K = number of output units/classes

Recall that in neural networks, we may have many output nodes. We denote _hΘ(x)k_ as being a hypothesis that results in the 
_kth_ output. Our cost function for neural networks is going to be a generalization of the one we used for logistic regression. Recall that the cost function for regularized logistic regression was:

![alt text](/Week_5/NeuralNetworksLearning/Assets/1.png)

For neural networks, it is going to be slightly more complicated:

![alt text](/Week_5/NeuralNetworksLearning/Assets/2.png)

We have added a few nested summations to account for our multiple output nodes. In the first part of the equation, before the square brackets, we have an additional nested summation that loops through the number of output nodes.

In the regularization part, after the square brackets, we must account for multiple theta matrices. The number of columns in our current theta matrix is equal to the number of nodes in our current layer (including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

Note:

* the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
* the triple sum simply adds up the squares of all the individual Θs in the entire network.
* the i in the triple sum does __not__ refer to training example i

# Backpropagation Algorithm

"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. Our goal is to compute:

_minΘ J(Θ)_

That is, we want to minimize our cost function J using an optimal set of parameters in theta. In this section we'll look at the equations we use to compute the partial derivative of J(Θ):

![alt text](/Week_5/NeuralNetworksLearning/Assets/3.png)

To do so, we use the following algorithm:

![alt text](/Week_5/NeuralNetworksLearning/Assets/4.png)

__Back propagation Algorithm__

Given training set _{(x(1),y(1))...(x(m),y(m))}_

* Set _Δi,j(l) := 0 for all (l,i,j)_, (hence you end up having a matrix full of zeros)

For training example t = 1 to m:

1. Set _a(1) := x(t)_
2. Perform forward propagation to compute _a(l)_ for _l = 2,3,...,L_

![alt text](/Week_5/NeuralNetworksLearning/Assets/5.png)

3. Using _y(t)_, compute _δ(L) = a(L) − y(t)_

Where L is our total number of layers and a(L) is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

4. Compute _δ(L−1),δ(L−2),...,δ(2)_ using _δ(l) = ((Θ(l))'δ(l+1)). ∗ a(l). ∗ (1−a(l))_

The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by _z(l)_.

The g-prime derivative terms can also be written out as:

_g′(z(l)) = a(l). ∗ (1 − a(l))_

5. _Δi,j(l) := Δi,j(l) + aj(l) δi(l+1)_ or with vectorization, _Δ(l) := Δ(l) + δ(l+1)(a(l))'_

Hence we update our new Δ matrix.

![alt text](/Week_5/NeuralNetworksLearning/Assets/10.png)

The capital delta matrix _D_ is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get:

![alt text](/Week_5/NeuralNetworksLearning/Assets/11.png)

# Backpropagation Intuition

Recall that the cost function for a neural network is:

![alt text](/Week_5/NeuralNetworksLearning/Assets/6.png)

If we consider simple non-multiclass classification _(k = 1)_ and disregard regularization, the cost is computed with:

![alt text](/Week_5/NeuralNetworksLearning/Assets/7.png)

Intuitively, _δj(l)_ is the "error" for _aj(l)_ (unit j in layer l). More formally, the delta values are actually the derivative of the cost function:

![alt text](/Week_5/NeuralNetworksLearning/Assets/8.png)

Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope the more incorrect we are. Let us consider the following neural network below and see how we could calculate some _δj(l)_:

![alt text](/Week_5/NeuralNetworksLearning/Assets/9.png)

In the image above, to calculate _δ2(2)_, we multiply the weights _Θ12(2)_ and _Θ22(2)_ by their respective _δ_ values found to the right of each edge. So we get _δ2(2) = Θ12(2) * δ1(3) + Θ22(2) * δ2(3)_. To calculate every single possible _δj(l)_, we could start from the right of our diagram. We can think of our edges as our _Θij_. Going from right to left, to calculate the value of _δj(l)_, you can just take the over all sum of each weight times the _δ_ it is coming from. Hence, another example would be _δ2(3) = Θ12(3) * δ1(4)_.
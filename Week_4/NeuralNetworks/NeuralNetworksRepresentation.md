# Model Representation I

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (__dendrites__) as electrical inputs (called "spikes") that are channeled to outputs (__axons__). In our model, our dendrites are like the input features _x1...xn_, and the output is the result of our hypothesis function. In this model our _x0_ input node is sometimes called the "bias unit." It is always equal to 1. In neural networks, we use the same logistic function as in classification, 
_1 / 1 + e^−θTx_, yet we sometimes call it a sigmoid (logistic) __activation__ function. In this situation, our "theta" parameters are sometimes called "weights".

Visually, a simplistic representation looks like:

![alt text](/Week_4/NeuralNetworks/Assets/1.png)

Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers."

In this example, we label these intermediate or "hidden" layer nodes _a(2)0...a(2)n_ and call them "activation units."

__a(j)i__ = "activation" of unit _i_ in layer _j_

__Θ(j)__ = matrix of weights controlling function mapping from layer _j_ to layer _j+1_

If we had one hidden layer, it would look like:

![alt text](/Week_4/NeuralNetworks/Assets/2.png)

The values for each of the "activation" nodes is obtained as follows:

![alt text](/Week_4/NeuralNetworks/Assets/3.png)

This is saying that we compute our activation nodes by using a 3×4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix _Θ(2)_ containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, _Θ(j)Θ_.

The dimensions of these matrices of weights is determined as follows:

If network has _sj_ units in layer _j_ and _sj+1_ units in layer _j+1_, then _Θ(j)_ will be of dimension _sj+1 × (sj+1)_.

The +1 comes from the addition in _Θ(j)_ of the "bias nodes," _x0_ and _Θ0(j)_. In other words the output nodes will not include the bias nodes while the inputs will. The following image summarizes our model representation:

![alt text](/Week_4/NeuralNetworks/Assets/4.png)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of 
_Θ(1)_ is going to be 4×3 where _sj = 2_ and _sj+1 = 4_, so _sj+1 × (sj+1) = 4×3_.

# Model Representation II

To re-iterate, the following is an example of a neural network:

![alt text](/Week_4/NeuralNetworks/Assets/3.png)

In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable _zk(j)_ that encompasses the parameters inside our _g_ function. In our previous example if we replaced by the variable z for all the parameters we would get:

![alt text](/Week_4/NeuralNetworks/Assets/5.png)

In other words, for layer _j = 2_ and node _k_, the variable _z_ will be:

![alt text](/Week_4/NeuralNetworks/Assets/6.png)

The vector representation of _x_ and _zj_ is:

![alt text](/Week_4/NeuralNetworks/Assets/7.png)

Setting _x = a(1)_, we can rewrite the equation as:

![alt text](/Week_4/NeuralNetworks/Assets/8.png)

We are multiplying our matrix _Θ(j−1)_ with dimensions _sj × (n+1)_(where sj is the number of our activation nodes) by our vector _a(j−1)_ with height _(n+1)_. This gives us our vector _z(j)_ with height _sj_. Now we can get a vector of our activation nodes for layer _j_ as follows:

_a(j) = g(z(j))_

Where our function g can be applied element-wise to our vector _z(j)_.

We can then add a bias unit (equal to 1) to layer j after we have computed _a(j)_. This will be element _a0(j)_ and will be equal to 1. To compute our final hypothesis, let's first compute another _z_ vector:

_z(j+1) = Θ(j)a(j)_

We get this final z vector by multiplying the next theta matrix after _Θ(j−1)_ with the values of all the activation nodes we just got. This last theta matrix _Θ(j)_ will have only one row which is multiplied by one column _a(j)_ so that our result is a single number. We then get our final result with:

_hΘ(x) = a(j+1) = g(z(j+1))_

Notice that in this last step, between layer _j_ and layer _j+1_, we are doing exactly the same thing as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.
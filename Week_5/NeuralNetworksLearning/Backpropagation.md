# Backpropagation

__Implementation Note: Unrolling Parameters__

With neural networks, we are working with sets of matrices:

Θ(1),Θ(2),Θ(3),...

D(1),D(2),D(3),...

In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

	thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
	deltaVector = [ D1(:); D2(:); D3(:) ]

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11, then we can get back our original matrices from the "unrolled" versions as follows:

	Theta1 = reshape(thetaVector(1:110),10,11)
	Theta2 = reshape(thetaVector(111:220),10,11)
	Theta3 = reshape(thetaVector(221:231),1,11)

To summarize:

![alt text](/Week_5/NeuralNetworksLearning/Assets/12.png)

# Gradient Checking

Gradient checking will assure that our backpropagation works as intended. We can approximate the derivative of our cost function with:

![alt text](/Week_5/NeuralNetworksLearning/Assets/13.png)

With multiple theta matrices, we can approximate the derivative with respect to Θj as follows:

![alt text](/Week_5/NeuralNetworksLearning/Assets/14.png)

A small value for _ϵ_ (epsilon) such as _ϵ = 10^−4_, guarantees that the math works out properly. If the value for _ϵ_ is too small, we can end up with numerical problems.

Hence, we are only adding or subtracting epsilon to the _Θj_ matrix. In octave we can do it as follows:

	epsilon = 1e-4;
	for i = 1:n,
  		thetaPlus = theta;
  		thetaPlus(i) += epsilon;
  		thetaMinus = theta;
  		thetaMinus(i) -= epsilon;
  		gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
	end;

We previously saw how to calculate the deltaVector. So once we compute our gradApprox vector, we can check that gradApprox ≈ deltaVector.

Once you have verified __once__ that your backpropagation algorithm is correct, you don't need to compute gradApprox again. The code to compute gradApprox can be very slow.

# Random Initialization

Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly. Instead we can randomly initialize our weights for our _Θ_ matrices using the following method:

![alt text](/Week_5/NeuralNetworksLearning/Assets/15.png)

Hence, we initialize each _Θij(l)_ to a random value between [−ϵ,ϵ]. Using the above formula guarantees that we get the desired bound. The same procedure applies to all the _Θ_'s. Below is some working code you could use to experiment.

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

	Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
	Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
	Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;

rand(x,y) is just a function in octave that will initialize a matrix of random real numbers between 0 and 1.
(Note: the epsilon used above is unrelated to the epsilon from Gradient Checking).

# Sum up

First, pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.

* Number of input units = dimension of features _x(i)_
* Number of output units = number of classes
* Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
* Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.

__Training a Neural Network__

* Randomly initialize the weights
* Implement forward propagation to get _hΘ(x(i))_ for any x(i)
* Implement the cost function
* Implement backpropagation to compute partial derivatives
* Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
* Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.
* When we perform forward and back propagation, we loop on every training example:


	for i = 1:m,
		Perform forward propagation and backpropagation using example (x(i),y(i))
   		(Get activations a(l) and delta terms d(l) for l = 2,...,L

The following image gives us an intuition of what is happening as we are implementing our neural network:

![alt text](/Week_5/NeuralNetworksLearning/Assets/16.png)

Ideally, you want _hΘ(x(i) ≈ y(i)_. This will minimize our cost function. However, keep in mind that _J(Θ)_ is not convex and thus we can end up in a local minimum instead.

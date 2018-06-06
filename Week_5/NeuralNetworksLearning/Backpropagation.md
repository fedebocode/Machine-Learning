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
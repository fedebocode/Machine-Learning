# One-vs-All Classification Example

In the previous part of this exercise, you implemented multi-class logistic re- gression to recognize handwritten digits. However, logistic regression cannot form more complex hypotheses as it is only a linear classifier.

You will implement a neural network to rec- ognize handwritten digits using the same training set as before. The neural network will be able to represent complex models that form non-linear hy- potheses. You will be using parameters from a neural network that has been already trained. Your goal is to implement the feedforward propagation algorithm to use our weights for prediction.

Our neural network is shown in the image below:

![alt text](/Week_4/Octave_MatlabTutorials/Assets/2.png)

It has 3 layers, an input layer, a hidden layer and an output layer. Recall that our inputs are pixel values of digit images. Since the images are of size 20×20, this gives us 400 input layer units (excluding the extra bias unit which always outputs +1). As before, the training data will be loaded into the variables _X_ and _y_.
You have been provided with a set of network parameters (_Θ(1)_,_Θ(2)_) already trained. These are stored in ex3weights.mat and will be loaded by _ex3 nn.m_ into _Theta1_ and _Theta2_ The parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

Initialize the model:

	clear ; close all; clc

Setup the parameters:

	input_layer_size  = 400;  			// 20x20 Input Images of Digits
	hidden_layer_size = 25;   			// 25 hidden units
	num_labels = 10;          			// 10 labels, from 1 to 10 (mapped "0" to label 10)

Load Training Data:

	load('ex3data1.mat');
	m = size(X, 1);

Randomly select 100 data points to display:

	sel = randperm(size(X, 1));
	sel = sel(1:100);

	displayData(X(sel, :));

Load the weights into variables Theta1 and Theta2:

	load('ex3weights.mat');

Predict labels of training set:

	pred = predict(Theta1, Theta2, X);
	fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

To give you an idea of the network's output, you can also run through the examples one at the a time to see what it is predicting.

Randomly permute examples:
	
	rp = randperm(m);

	for i = 1:m
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      	break
    	end
	end

![alt text](/Week_4/Octave_MatlabTutorials/Assets/3.png)
![alt text](/Week_4/Octave_MatlabTutorials/Assets/4.png)

__Predict__

	function p = predict(Theta1, Theta2, X)

	m = size(X, 1);
	num_labels = size(Theta2, 1);

	p = zeros(size(X, 1), 1);

	a1 = [ones(m, 1) X];
	z2 = a1 * Theta1';
	a2 = [ones(size(z2), 1) sigmoid(z2)];
	z3 = a2 * Theta2';
	a3 = sigmoid(z3);

	[val, p] = max(a3, [], 2);

	end


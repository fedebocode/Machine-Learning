# Neural Network Learning Example

In this exercise, you will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition.
In the previous exercise, you implemented feedforward propagation for neu- ral networks and used it to predict handwritten digits with the weights we provided. In this exercise, you will implement the backpropagation algorithm to learn the parameters for the neural network.

Initialize model:

    clear ; close all; clc

Setup the parameters:

    input_layer_size  = 400;  		//20x20 Input Images of Digits
    hidden_layer_size = 25;   		//25 hidden units
    num_labels = 10;          		//10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

Load Training Data:

    load('ex4data1.mat');
    m = size(X, 1);

Randomly select 100 data points to display:

    sel = randperm(size(X, 1));
    sel = sel(1:100);

	 displayData(X(sel, :));

![alt text](/Week_5/Octave_MatlabTutorials/Assets/1.png)

Load the weights into variables Theta1 and Theta2:

    load('ex4weights.mat');

Unroll parameters:

    nn_params = [Theta1(:) ; Theta2(:)];

Weight regularization parameter (we set this to 0 here).

    lambda = 0;
    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

Weight regularization parameter (we set this to 1 here).

    lambda = 1;
    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

    g = sigmoidGradient([-1 -0.5 0 0.5 1]);

Initialize weights:

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

Unroll parameters:

      initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

Check gradients:

    checkNNGradients;

Check gradients by running checkNNGradients:

    lambda = 3;
    checkNNGradients(lambda);

Output the costFunction debugging values:

    debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

Train neural network using "fmincg" function. Recall that these advanced optimizers are able to train our cost functions efficiently as long as we provide them with the gradient computations.

    options = optimset('MaxIter', 50);

You should also try different values of lambda:

    lambda = 1;

Create "short hand" for the cost function to be minimized:
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

Now, costFunction is a function that takes in only one argument (the neural network parameters):

    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Obtain Theta1 and Theta2 back from nn_params:

    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

You can now "visualize" what the neural network is learning by displaying the hidden units to see what features they are capturing in the data.

    displayData(Theta1(:, 2:end));

![alt text](/Week_5/Octave_MatlabTutorials/Assets/2.png)

Predict:

    pred = predict(Theta1, Theta2, X);
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

## Functions

__Neural Network Cost Function__

    function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

      Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
      Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

      m = size(X, 1);
         

      J = 0;
      Theta1_grad = zeros(size(Theta1));
      Theta2_grad = zeros(size(Theta2));

      X = [ones(m, 1) X];

      % Activations for each layer
      for i=1:m
        a1 = X(i,:)';
        z2 = Theta1 * a1;
        a2 = [1; sigmoid(z2)];
        z3 = Theta2 * a2;
        a3 = sigmoid(z3);
        h = a3;                   //final layer activation is output vector

        yVec = (1:num_labels)' == y(i);
        J = J + sum(-yVec .* log(h) - (1 - yVec) .* log(1 - h));

      % Backpropagation:
        delta3 = a3 - yVec;
        delta2 = Theta2' * delta3 .* (a2 .* (1 - a2));
        Theta2_grad = Theta2_grad + delta3 * a2';
        Theta1_grad = Theta1_grad + delta2(2:end) * a1';
      end;

Scaling cost function and gradients:
    
      J = J / m;
      Theta1_grad = Theta1_grad / m;
      Theta2_grad = Theta2_grad / m;

Regularization:

      J = J + (lambda / (2 * m)) * (sumsq(Theta1(:, 2:end)(:)) + sumsq(Theta2(:, 2:end)(:)));
      Theta1_grad = Theta1_grad + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
      Theta2_grad = Theta2_grad + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

Unroll gradients:
    
      grad = [Theta1_grad(:) ; Theta2_grad(:)];

    end

__Sigmoid Gradient__

    function g = sigmoidGradient(z)
      g = zeros(size(z));
      g = sigmoid(z) .* (1 - sigmoid(z));
    end

__Initialize Random Weights__

    function W = randInitializeWeights(L_in, L_out)

      W = zeros(L_out, 1 + L_in);
      epsilon_init = 0.12;
      W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
    end

__Compute Numerical Gradient__

    function numgrad = computeNumericalGradient(J, theta)

      numgrad = zeros(size(theta));
      perturb = zeros(size(theta));
      e = 1e-4;
      for p = 1:numel(theta)
        perturb(p) = e;
        loss1 = J(theta - perturb);
        loss2 = J(theta + perturb);
        numgrad(p) = (loss2 - loss1) / (2*e);
        perturb(p) = 0;
      end
    end

__Check Gradients__

    function checkNNGradients(lambda)

      if ~exist('lambda', 'var') || isempty(lambda)
          lambda = 0;
      end

      input_layer_size = 3;
      hidden_layer_size = 5;
      num_labels = 3;
      m = 5;

      Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
      Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);

      X  = debugInitializeWeights(m, input_layer_size - 1);
      y  = 1 + mod(1:m, num_labels)';

Unroll parameters:

      nn_params = [Theta1(:) ; Theta2(:)];

      costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

      [cost, grad] = costFunc(nn_params);
      numgrad = computeNumericalGradient(costFunc, nn_params);

Visually examine the two gradient computations. The two columns you get should be very similar. 

      disp([numgrad grad]);

Evaluate the norm of the difference between two solutions. If you have a correct implementation, and assuming you used EPSILON = 0.0001 in computeNumericalGradient.m, then diff below should be less than 1e-9

      diff = norm(numgrad-grad)/norm(numgrad+grad);

      fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

    end

__Debug Initialized Weights__

    function W = debugInitializeWeights(fan_out, fan_in)

      W = zeros(fan_out, 1 + fan_in);
      W = reshape(sin(1:numel(W)), size(W)) / 10;
    end

__Sigmoid__

    function g = sigmoid(z)

      g = 1.0 ./ (1.0 + exp(-z));
    end

__Predict__

    function p = predict(Theta1, Theta2, X)

      m = size(X, 1);
      num_labels = size(Theta2, 1);

      p = zeros(size(X, 1), 1);

      h1 = sigmoid([ones(m, 1) X] * Theta1');
      h2 = sigmoid([ones(m, 1) h1] * Theta2');
      [dummy, p] = max(h2, [], 2);
    end
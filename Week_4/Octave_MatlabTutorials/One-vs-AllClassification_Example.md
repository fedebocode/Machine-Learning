# One-vs-All Classification Example

In this exercise, you will implement One-vs-All logistic regression to recognize hand-written digits.
Automated handwritten digit recognition is widely used today, from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks.
You will extend your previous implemen- tion of logistic regression and apply it to One-vs-All classification.

You are given a data set in _ex3data1.mat_ that contains 5000 training examples of handwritten digits.

There are 5000 training examples in _ex3data1.mat_, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400 dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix _X_ where every row is a training example for a handwritten digit image.

The second part of the training set is a 5000 dimensional vector _y_ that contains labels for the training set.

Initialize model:

	clear ; close all; clc

Setup the parameters:

	input_layer_size  = 400;  		// 20x20 Input Images of Digits
	num_labels = 10;          		// 10 labels, from 1 to 10 (mapped "0" to label 10)

Load Training Data:

	load('ex3data1.mat'); 			// training data stored in arrays X, y
	m = size(X, 1);

Randomly select 100 data points to display:

	rand_indices = randperm(m);
	sel = X(rand_indices(1:100), :);

	displayData(sel);

![alt text](/Week_4/Octave_MatlabTutorials/Assets/1.png)

Set theta and compute logistic regression:

	theta_t = [-2; -1; 1; 2];
	X_t = [ones(5,1) reshape(1:15,5,3)/10];
	y_t = ([1;0;1;0;1] >= 0.5);
	lambda_t = 3;
	[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

	lambda = 0.1;
	[all_theta] = oneVsAll(X, y, num_labels, lambda);

Predict for One Vs All:

	pred = predictOneVsAll(all_theta, X);

	fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

__Display Data:__

	function [h, display_array] = displayData(X, example_width)

Set example_width automatically if not passed in:

	if ~exist('example_width', 'var') || isempty(example_width) 
		example_width = round(sqrt(size(X, 2)));
	end

Gray Image:
	
	colormap(gray);

Compute rows, cols:

	[m n] = size(X);
	example_height = (n / example_width);

Compute number of items to display:

	display_rows = floor(sqrt(m));
	display_cols = ceil(m / display_rows);

Between images padding:

	pad = 1;

Setup blank display:

	display_array = - ones(pad + display_rows * (example_height + pad), ...
                       	pad + display_cols * (example_width + pad));

Copy each example into a patch on the display array:

	curr_ex = 1;
	for j = 1:display_rows
		for i = 1:display_cols
			if curr_ex > m, 
				break; 
			end

			max_val = max(abs(X(curr_ex, :)));
			display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              	pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
							reshape(X(curr_ex, :), example_height, example_width) / max_val;
			curr_ex = curr_ex + 1;
		end
		if curr_ex > m, 
			break; 
		end
	end

Display Image:

	h = imagesc(display_array, [-1 1]);
	axis image off
	drawnow;

	end

__Logistic Regression Cost Function__

	function [J, grad] = lrCostFunction(theta, X, y, lambda)

	m = length(y); 

	J = 0;
	grad = zeros(size(theta));

	h = sigmoid(X * theta);
	cost = sum(-y .* log(h) - (1 - y) .* log(1 - h));
	grad = X' * (h - y);

	grad_reg = lambda * theta;
	grad_reg(1) = 0;
	grad = grad + grad_reg;

	J = cost / m + (lambda / (2.0 * m)) * sum(theta(2:size(theta)) .^ 2);
	grad = grad / m;

	grad = grad(:);

	end

__One-vs-All__

	function [all_theta] = oneVsAll(X, y, num_labels, lambda)

	m = size(X, 1);
	n = size(X, 2);

	all_theta = zeros(num_labels, n + 1);

	X = [ones(m, 1) X];

	for c = 1:num_labels
     	initial_theta = zeros(n + 1, 1);
     	options = optimset('GradObj', 'on', 'MaxIter', 50);
     	[theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
     	all_theta(c, :) = theta';
	end;
	end

__Predict One-vs-All__

	function p = predictOneVsAll(all_theta, X)

	m = size(X, 1);
	num_labels = size(all_theta, 1);

	p = zeros(size(X, 1), 1);

	X = [ones(m, 1) X];

	[m, p] = max(sigmoid(X * all_theta'), [], 2);

	end

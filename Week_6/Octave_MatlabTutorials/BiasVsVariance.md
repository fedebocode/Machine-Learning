# Regularized Linear Regression and Bias vs Variance

In this exercise, you will implement regularized linear regression and use it to study models with different bias-variance properties.

In the first half of the exercise, you will implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir. In the next half, you will go through some diagnostics of debugging learning algorithms and examine the effects of bias v.s. variance.

We will begin by visualizing the dataset containing historical records on the change in the water level, _x_, and the amount of water flowing out of the dam, _y_.

This dataset is divided into three parts:

* A __training__ set that your model will learn on: _X_, _y_
* A __cross validation set__ for determining the regularization parameter: _Xval_, _yval_
* A __test__ set for evaluating performance. These are “unseen” examples which your model did not see during training: _Xtest_, _ytest_

In the following parts, you will implement linear regression and use that to fit a straight line to the data and plot learning curves. Following that, you will implement polynomial regression to find a better fit to the data.

Initialization:

	clear ; close all; clc

Load Training Data:

	fprintf('Loading and Visualizing Data ...\n')

Load from ex5data1: 
	
	load ('ex5data1.mat');
	m = size(X, 1);

Plot training data:

	plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
	xlabel('Change in water level (x)');
	ylabel('Water flowing out of the dam (y)');

![alt text](/Week_6/Octave_MatlabTutorials/Assets/PlotData.png)

Implement the cost function for regularized linear regression:

	theta = [1 ; 1];
	J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

Implement the gradient for regularized linear regression:

	theta = [1 ; 1];
	[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

Train Linear Regression:

	lambda = 0;
	[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

Plot fit over the data:

	plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
	xlabel('Change in water level (x)');
	ylabel('Water flowing out of the dam (y)');
	hold on;
	plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
	hold off;

![alt text](/Week_6/Octave_MatlabTutorials/Assets/PlotFitData.png)

Implement the learningCurve function:

	lambda = 0;
	[error_train, error_val] = learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

	plot(1:m, error_train, 1:m, error_val);
	title('Learning curve for linear regression')
	legend('Train', 'Cross Validation')
	xlabel('Number of training examples')
	ylabel('Error')
	axis([0 13 0 150])

![alt text](/Week_6/Octave_MatlabTutorials/Assets/LearningCurveLinearRegression.png)

One solution to this is to use polynomial regression. Map each example into its powers:

	p = 8;
	X_poly = polyFeatures(X, p);
	[X_poly, mu, sigma] = featureNormalize(X_poly);
	X_poly = [ones(m, 1), X_poly];  

	// Map X_poly_test and normalize (using mu and sigma)
	X_poly_test = polyFeatures(Xtest, p);
	X_poly_test = bsxfun(@minus, X_poly_test, mu);
	X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
	X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];

	// Map X_poly_val and normalize (using mu and sigma)
	X_poly_val = polyFeatures(Xval, p);
	X_poly_val = bsxfun(@minus, X_poly_val, mu);
	X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
	X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];

Learning Curve for Polynomial Regression:

	lambda = 0;
	[theta] = trainLinearReg(X_poly, y, lambda);

Plot training data and fit:

	figure(1);
	plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
	plotFit(min(X), max(X), mu, sigma, theta, p);
	xlabel('Change in water level (x)');
	ylabel('Water flowing out of the dam (y)');
	title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

![alt text](/Week_6/Octave_MatlabTutorials/Assets/PolynomialRegressionFit.png)

	figure(2);
	[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
	plot(1:m, error_train, 1:m, error_val);

	title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
	xlabel('Number of training examples')
	ylabel('Error')
	axis([0 13 0 100])
	legend('Train', 'Cross Validation')

![alt text](/Week_6/Octave_MatlabTutorials/Assets/PolynomialRegressionLearningCurve.png)

Implement validationCurve to test various values of lambda on a validation set. You will then use this to select the "best" lambda value.

	[lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval);

	close all;
	plot(lambda_vec, error_train, lambda_vec, error_val);
	legend('Train', 'Cross Validation');
	xlabel('lambda');
	ylabel('Error');

![alt text](/Week_6/Octave_MatlabTutorials/Assets/TrainCrossValidation.png)

Test different values of the regularization parameter lambda and get to observe how it affects the bias-variance of regularized polynomial regression.

__Lambda = 1__

![alt text](/Week_6/Octave_MatlabTutorials/Assets/PolynomialRegressionFit_Lambda1.png)
![alt text](/Week_6/Octave_MatlabTutorials/Assets/PolynomialRegressionLearningCurve_Lambda1.png)

__Lambda = 500__

![alt text](/Week_6/Octave_MatlabTutorials/Assets/PolynomialRegressionFit_Lambda500.png)
![alt text](/Week_6/Octave_MatlabTutorials/Assets/PolynomialRegressionLearningCurve_Lambda500.png)

__Lambda = 1000__

![alt text](/Week_6/Octave_MatlabTutorials/Assets/PolynomialRegressionFit_Lambda1000.png)
![alt text](/Week_6/Octave_MatlabTutorials/Assets/PolynomialRegressionLearningCurve_Lambda1000.png)

### Functions

__Linear Regression Cost Function__

	function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

		m = length(y); % number of training examples
		J = 0;
		grad = zeros(size(theta));

		h = X * theta;
		errors = h - y;
		J = sumsq(errors) / (2 * m);
		grad = (1 / m) * X' * errors;

		% Regularization
		J = J + (lambda / (2 * m)) * sumsq(theta(2:end));
		grad = grad + (lambda / m) * [0; theta(2:end)];

		grad = grad(:);
	end

__Train Linear Regression__

	function [theta] = trainLinearReg(X, y, lambda)

		% Initialize Theta
		initial_theta = zeros(size(X, 2), 1); 

		% Create "short hand" for the cost function to be minimized
		costFunction = @(t) linearRegCostFunction(X, y, t, lambda);

		% Now, costFunction is a function that takes in only one argument
		options = optimset('MaxIter', 200, 'GradObj', 'on');

		% Minimize using fmincg
		theta = fmincg(costFunction, initial_theta, options);
	end

__Plot Fit__

	function plotFit(min_x, max_x, mu, sigma, theta, p)

		% Hold on to the current figure
		hold on;

		% We plot a range slightly bigger than the min and max values to get
		% an idea of how the fit will vary outside the range of the data points
		x = (min_x - 15: 0.05 : max_x + 25)';

		% Map the X values 
		X_poly = polyFeatures(x, p);
		X_poly = bsxfun(@minus, X_poly, mu);
		X_poly = bsxfun(@rdivide, X_poly, sigma);

		% Add ones
		X_poly = [ones(size(x, 1), 1) X_poly];

		% Plot
		plot(x, X_poly * theta, '--', 'LineWidth', 2)

		% Hold off to the current figure
		hold off
	end

__Feature Normalize__

	function [X_norm, mu, sigma] = featureNormalize(X)

		mu = mean(X);
		X_norm = bsxfun(@minus, X, mu);

		sigma = std(X_norm);
		X_norm = bsxfun(@rdivide, X_norm, sigma);
	end

__Poly Features__

	function [X_poly] = polyFeatures(X, p)

		% You need to return the following variables correctly.
		X_poly = zeros(numel(X), p);

		X_poly = X;
		for i=2:p
    		X_poly = [X_poly X.^i];
		end;
	end

__Validation Curve__

	function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)

		% Selected values of lambda (you should not change this)
		lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

		error_train = zeros(length(lambda_vec), 1);
		error_val = zeros(length(lambda_vec), 1);

		for i = 1:length(lambda_vec)
    		lambda = lambda_vec(i);
    		theta = trainLinearReg(X, y, lambda);
    		error_train(i) = linearRegCostFunction(X, y, theta, 0);
    		error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
		end;
	end
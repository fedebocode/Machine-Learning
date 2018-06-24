# Support Vector Machines Example

In this exercise, you will be using support vector machines (SVMs) with various example 2D datasets. Experimenting with these datasets will help you gain an intuition of how SVMs work and how to use a Gaussian kernel with SVMs. In the next half of the exercise, you will be using support vector machines to build a spam classifier.

Initialization:

	clear ; close all; clc

Load from ex6data1: 

	load('ex6data1.mat');
	plotData(X, y);

![alt text](/Week_7/Octave_MatlabTutorials/Assets/PlotData.png)

Training Linear SVM:

	C = 1;
	model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
	visualizeBoundaryLinear(X, y, model);

![alt text](/Week_7/Octave_MatlabTutorials/Assets/DecisionBoundaryLinear.png)

Implementing Gaussian Kernel:

	x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
	sim = gaussianKernel(x1, x2, sigma);

Training SVM with RBF Kernel:

	load('ex6data2.mat');
	plotData(X, y);

![alt text](/Week_7/Octave_MatlabTutorials/Assets/PlotData2.png)

	% SVM Parameters
	C = 1; sigma = 0.1;

	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
	visualizeBoundary(X, y, model);

![alt text](/Week_7/Octave_MatlabTutorials/Assets/DecisionBoundary.png)

Load next dataset:

	load('ex6data3.mat');
	plotData(X, y);

![alt text](/Week_7/Octave_MatlabTutorials/Assets/PlotData3.png)

	[C, sigma] = dataset3Params(X, y, Xval, yval);

Train the SVM:

	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
	visualizeBoundary(X, y, model);

![alt text](/Week_7/Octave_MatlabTutorials/Assets/DecisionBoundary2.png)

### Functions

__Gaussian Kernel__

	function sim = gaussianKernel(x1, x2, sigma)

		% Ensure that x1 and x2 are column vectors
		x1 = x1(:); x2 = x2(:);

		sim = 0;
		sim = exp(-sumsq(x1 - x2) / (2 * sigma^2));
	end

__Dataset3Params__

	function [C, sigma] = dataset3Params(X, y, Xval, yval)

		allC = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
		allSigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
		bestC = allC(1);
		bestSigma = allSigma(1);
		previousErr = 1000;

		for i=1:length(allC)
    		currentC = allC(i);
    		for j=1:length(allSigma)
        	currentSigma = allSigma(j);
        	model = svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
        	predictions = svmPredict(model, Xval);
        	err = mean(double(predictions ~= yval));
        	if err < previousErr
            	bestC = currentC;
            	bestSigma = currentSigma;
            	previousErr = err;
        		end;
    		end;
		end;

		C = bestC;
		sigma = bestSigma;

	end

__Email Features__

	function x = emailFeatures(word_indices)
 
		% Total number of words in the dictionary
		n = 1899;

		% You need to return the following variables correctly.
		x = zeros(n, 1);

		for i=1:length(word_indices)
    		idx = word_indices(i);
    		x(idx) = 1;
		end;
	end

__Plot Data__

	function plotData(X, y)

		% Find Indices of Positive and Negative Examples
		pos = find(y == 1); 
		neg = find(y == 0);

		% Plot Examples
		plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7)
		hold on;
		plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
		hold off;
	end

__Visualize Boundary__

	function visualizeBoundary(X, y, model, varargin)

		% Plot the training data on top of the boundary
		plotData(X, y)

		% Make classification predictions over a grid of values
		x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
		x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
		[X1, X2] = meshgrid(x1plot, x2plot);
		vals = zeros(size(X1));
		for i = 1:size(X1, 2)
   			this_X = [X1(:, i), X2(:, i)];
   			vals(:, i) = svmPredict(model, this_X);
		end

		% Plot the SVM boundary
		hold on
		contour(X1, X2, vals, [0.5 0.5], 'b');
		hold off;
	end

__Visualize Boundary Linear__

	function visualizeBoundaryLinear(X, y, model)

		w = model.w;
		b = model.b;
		xp = linspace(min(X(:,1)), max(X(:,1)), 100);
		yp = - (w(1)*xp + b)/w(2);
		plotData(X, y);
		hold on;
		plot(xp, yp, '-b'); 
		hold off
	end

__SVM Train, SMV Predict, porterStemmer, readFile, linearKernel, getVocabList__ included the repository folder.
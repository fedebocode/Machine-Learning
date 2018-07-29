# Anomaly Detection Example

We start this exercise by using a small dataset that is easy to visualize.

Our example case consists of 2 network server statistics across several machines: the latency and throughput of each machine.

This exercise will help us find possibly faulty (or very fast) machines.

Load dataset:

	load('ex8data1.mat');

Visualize the example dataset:

	plot(X(:, 1), X(:, 2), 'bx');
	axis([0 30 0 30]);
	xlabel('Latency (ms)');
	ylabel('Throughput (mb/s)');

![alt text](/Week_9/Octave_MatlabTutorials/Assets/PlotData.png)

We assume a Gaussian distribution for the dataset.

We first estimate the parameters of our assumed Gaussian distribution, then compute the probabilities for each of the points and then visualize both the overall distribution and where each of the points falls in terms of that distribution.

![alt text](/Week_9/Octave_MatlabTutorials/Assets/GaussianFit.png)

Estimate mu and sigma2:

	[mu sigma2] = estimateGaussian(X);

Returns the density of the multivariate normal at each data point (row) of X:

	p = multivariateGaussian(X, mu, sigma2);

Visualize the fit:

	visualizeFit(X,  mu, sigma2);
	xlabel('Latency (ms)');
	ylabel('Throughput (mb/s)');

![alt text](/Week_9/Octave_MatlabTutorials/Assets/AnomalyDetection.png)

Now you will find a good epsilon threshold using a cross-validation set probabilities given the estimated Gaussian distribution:

	pval = multivariateGaussian(Xval, mu, sigma2);
	[epsilon F1] = selectThreshold(yval, pval);

Find the outliers in the training set and plot theoutliers = find(p < epsilon);
Draw a red circle around those outliers:

	hold on
	plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
	hold off


We will now use the code from the previous part and apply it to a harder problem in which more features describe each datapoint and only some features indicate whether a point is an outlier.

Loads the second dataset. You should now have the variables X, Xval, yval in your environment:

	load('ex8data2.mat');

Apply the same steps to the larger dataset:

	[mu sigma2] = estimateGaussian(X);

Training set:

	p = multivariateGaussian(X, mu, sigma2);

Cross-validation set:

	pval = multivariateGaussian(Xval, mu, sigma2);

Find the best threshold:

	[epsilon F1] = selectThreshold(yval, pval);
    
### Functions

__Check Cost Function__

    function checkCostFunction(lambda)

        % Set lambda
        if ~exist('lambda', 'var') || isempty(lambda)
            lambda = 0;
        end

        %% Create small problem
        X_t = rand(4, 3);
        Theta_t = rand(5, 3);

        % Zap out most entries
        Y = X_t * Theta_t';
        Y(rand(size(Y)) > 0.5) = 0;
        R = zeros(size(Y));
        R(Y ~= 0) = 1;

        %% Run Gradient Checking
        X = randn(size(X_t));
        Theta = randn(size(Theta_t));
        num_users = size(Y, 2);
        num_movies = size(Y, 1);
        num_features = size(Theta_t, 2);

        numgrad = computeNumericalGradient( ...
        @(t) cofiCostFunc(t, Y, R, num_users, num_movies, ...
        num_features, lambda), [X(:); Theta(:)]);

        [cost, grad] = cofiCostFunc([X(:); Theta(:)],  Y, R, num_users, ...
        num_movies, num_features, lambda);

        disp([numgrad grad]);

    end
    
__Cofi Cost Func__

    function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
    num_features, lambda)

        % Unfold the U and W matrices from params
        X = reshape(params(1:num_movies*num_features), num_movies, num_features);
        Theta = reshape(params(num_movies*num_features+1:end), ...
        num_users, num_features);

        % You need to return the following values correctly
        J = 0;
        X_grad = zeros(size(X));
        Theta_grad = zeros(size(Theta));

        err = X * Theta' - Y;
        err = err .* R;
        J = 0.5 * sumsq(err(:));

        X_grad = err * Theta;
        Theta_grad = err' * X;

        % regularization
        J = J + (lambda / 2) * (sumsq(Theta(:)) + sumsq(X(:)));
        X_grad = X_grad + lambda * X;
        Theta_grad = Theta_grad + lambda * Theta;

        grad = [X_grad(:); Theta_grad(:)];

    end

__Compute Numerical Gradient__

    function numgrad = computeNumericalGradient(J, theta)
             
            numgrad = zeros(size(theta));
            perturb = zeros(size(theta));
            e = 1e-4;
            for p = 1:numel(theta)
                % Set perturbation vector
                perturb(p) = e;
                loss1 = J(theta - perturb);
                loss2 = J(theta + perturb);
                % Compute Numerical Gradient
                numgrad(p) = (loss2 - loss1) / (2*e);
                perturb(p) = 0;
            end
    end

__Estimate Gaussian__

    function [mu sigma2] = estimateGaussian(X)

        [m, n] = size(X);
        mu = zeros(n, 1);
        sigma2 = zeros(n, 1);

        mu = mean(X);
        sigma2 = var(X, 1);

    end
    
__Multivariate Gaussian__

    function p = multivariateGaussian(X, mu, Sigma2)

        k = length(mu);

        if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
            Sigma2 = diag(Sigma2);
        end

        X = bsxfun(@minus, X, mu(:)');
        p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
            exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

    end
    
__Normalize Ratings__

    function [Ynorm, Ymean] = normalizeRatings(Y, R)

        [m, n] = size(Y);
        Ymean = zeros(m, 1);
        Ynorm = zeros(size(Y));
        for i = 1:m
            idx = find(R(i, :) == 1);
            Ymean(i) = mean(Y(i, idx));
            Ynorm(i, idx) = Y(i, idx) - Ymean(i);
        end
    end

__Select Threshold__

    function [bestEpsilon bestF1] = selectThreshold(yval, pval)

        bestEpsilon = 0;
        bestF1 = 0;
        F1 = 0;

        stepsize = (max(pval) - min(pval)) / 1000;
        for epsilon = min(pval):stepsize:max(pval)
            predictions = pval < epsilon;
            true_positives = sum((predictions == 1) & (yval == 1));
            false_positives = sum((predictions == 1) & (yval == 0));
            false_negatives = sum((predictions == 0) & (yval == 1));
            precision = true_positives / (true_positives + false_positives);
            recall = true_positives / (true_positives + false_negatives);
            F1 = 2 * precision * recall / (precision + recall);

            if F1 > bestF1
                bestF1 = F1;
                bestEpsilon = epsilon;
            end
        end
    end

__Visualize Fit__

    function visualizeFit(X, mu, sigma2)

        [X1,X2] = meshgrid(0:.5:35); 
        Z = multivariateGaussian([X1(:) X2(:)],mu,sigma2);
        Z = reshape(Z,size(X1));

        plot(X(:, 1), X(:, 2),'bx');
        hold on;
        % Do not plot if there are infinities
        if (sum(isinf(Z)) == 0)
            contour(X1, X2, Z, 10.^(-20:3:0)');
        end
        hold off;
    end

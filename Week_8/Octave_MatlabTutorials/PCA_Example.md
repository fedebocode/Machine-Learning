# PCA Example

In this exercise, you will use principal component analysis (PCA) to perform dimensionality reduction. You will first experiment with an example 2D dataset to get intuition on how PCA works, and then use it on a bigger dataset of 5000 face image dataset.

Initialization:

	clear ; close all; clc

Load data:
	
	load ('ex7data1.mat');

Visualize the dataset:

	plot(X(:, 1), X(:, 2), 'bo');
	axis([0.5 6.5 2 8]); axis square;

![alt text](/Week_8/Octave_MatlabTutorials/Assets/Plot.png)

Normalize X:

	[X_norm, mu, sigma] = featureNormalize(X);

Run PCA:

	[U, S] = pca(X_norm);

Compute mu, the mean of the each feature:

Draw the eigenvectors centered at mean of data. These lines show the directions of maximum variations in the dataset:
	
	hold on;
	drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
	drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
	hold off;

Plot the normalized dataset (returned from pca):

	plot(X_norm(:, 1), X_norm(:, 2), 'bo');
	axis([-4 3 -4 3]); axis square

![alt text](/Week_8/Octave_MatlabTutorials/Assets/Eigen.png)

Project the data onto K = 1 dimension:

	K = 1;
	Z = projectData(X_norm, U, K);
	X_rec  = recoverData(Z, U, K);

Draw lines connecting the projected points to the original points:

	hold on;
	plot(X_rec(:, 1), X_rec(:, 2), 'ro');
	for i = 1:size(X_norm, 1)
    	drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
	end
	hold off

![alt text](/Week_8/Octave_MatlabTutorials/Assets/Projections.png)

Load Face dataset:

	load ('ex7faces.mat')

Display the first 100 faces in the dataset:

	displayData(X(1:100, :));

![alt text](/Week_8/Octave_MatlabTutorials/Assets/FacesDataset.png)

Before running PCA, it is important to first normalize X by subtracting the mean value from each feature:

	[X_norm, mu, sigma] = featureNormalize(X);

Run PCA:

	[U, S] = pca(X_norm);

Visualize the top 36 eigenvectors found:

	displayData(U(:, 1:36)');

![alt text](/Week_8/Octave_MatlabTutorials/Assets/PCA.png)

Project images to the eigen space using the top k eigenvectors:

	K = 100;
	Z = projectData(X_norm, U, K);

Project images to the eigen space using the top K eigen vectors and visualize only using those K dimensions

	K = 100;
	X_rec  = recoverData(Z, U, K);

Display normalized data:

	subplot(1, 2, 1);
	displayData(X_norm(1:100,:));
	title('Original faces');
	axis square;

Display reconstructed data from only k eigenfaces:

	subplot(1, 2, 2);
	displayData(X_rec(1:100,:));
	title('Recovered faces');
	axis square;

![alt text](/Week_8/Octave_MatlabTutorials/Assets/Recovered.png)

We first visualize this output in 3D, and then apply PCA to obtain a visualization in 2D:

	close all; close all; clc

% Reload the image from the previous exercise and run K-Means on it:

	A = double(imread('bird_small.png'));
	load ('bird_small.mat');
	A = A / 255;
	img_size = size(A);
	X = reshape(A, img_size(1) * img_size(2), 3);
	K = 16; 
	max_iters = 10;
	initial_centroids = kMeansInitCentroids(X, K);
	[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

Sample 1000 random indexes (since working with all the data is too expensive. If you have a fast computer, you may increase this:

	sel = floor(rand(1000, 1) * size(X, 1)) + 1;

Setup Color Palette:

	palette = hsv(K);
	colors = palette(idx(sel), :);

Visualize the data and centroid memberships in 3D:

	figure;
	scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
	title('Pixel dataset plotted in 3D. Color shows centroid memberships');

![alt text](/Week_8/Octave_MatlabTutorials/Assets/Plot3D.png)

Use PCA to project this cloud to 2D for visualization.
Subtract the mean to use PCA:

	[X_norm, mu, sigma] = featureNormalize(X);

PCA and project the data to 2D:

	[U, S] = pca(X_norm);
	Z = projectData(X_norm, U, 2);

Plot in 2D:

	figure;
	plotDataPoints(Z(sel, :), idx(sel), K);
	title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');

![alt text](/Week_8/Octave_MatlabTutorials/Assets/Plot2D.png)

### Functions

__Feature Normalize__

	function [X_norm, mu, sigma] = featureNormalize(X)

		mu = mean(X);
		X_norm = bsxfun(@minus, X, mu);

		sigma = std(X_norm);
		X_norm = bsxfun(@rdivide, X_norm, sigma);
	end

__Project Data__

	function Z = projectData(X, U, K)

		Z = zeros(size(X, 1), K);

		Ureduce = U(:, 1:K);

		for i=1:size(X, 1)
			z = Ureduce' * X(i, :)';
			Z(i, :) = z;
		end;
	end

__Recover Data__

	function X_rec = recoverData(Z, U, K)

		X_rec = zeros(size(Z, 1), size(U, 1));

		Ureduce = U(:, 1:K);

		for i=1:size(Z, 1)
			z = Z(i, :)';
			x = Ureduce * z;
			X_rec(i, :) = x';
		end;
	end

__PCA__

	function [U, S] = pca(X)

		[m, n] = size(X);

		U = zeros(n);
		S = zeros(n);

		Sigma = (1 / m) * X' * X;
		[U, S, V] = svd(Sigma);
	end
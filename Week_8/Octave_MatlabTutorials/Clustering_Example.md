# K-means Clustering Example

In this exercise, you will implement the K-means clustering algorithm and apply it to compress an image. In the second part, you will use principal component analysis to find a low-dimensional representation of face images. You wil use the K-means algorithm for image compression by reducing the number of colors that occur in an image to only those that are most common in that image.

The K-means algorithm is a method to automatically cluster similar data examples together. Concretely, you are given a training set _{x(1),...,x(m)}_ (where _x(i)_ ∈ _Rn_), and want to group the data into a few cohesive “clusters”. The intuition behind K-means is an iterative procedure that starts by guess- ing the initial centroids, and then refines this guess by repeatedly assigning examples to their closest centroids and then recomputing the centroids based on the assignments.

Initialization: 
	
	clear ; close all; clc

Load an example dataset that we will be using:

	load('ex7data2.mat');

Select an initial set of centroids:

	K = 3;
	initial_centroids = [3 3; 6 2; 8 5];

Find the closest centroids for the examples using the initial_centroids:

	idx = findClosestCentroids(X, initial_centroids);

Compute means based on the closest centroids:

	centroids = computeCentroids(X, idx, K);

Load an example dataset:
	
	load('ex7data2.mat');

Settings for running K-Means:

	K = 3;
	max_iters = 10;
	initial_centroids = [3 3; 6 2; 8 5];

Run K-Means algorithm:

	[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);

![alt text](/Week_8/Octave_MatlabTutorials/Assets/Iteration_1.png)
![alt text](/Week_8/Octave_MatlabTutorials/Assets/Iteration_2.png)
![alt text](/Week_8/Octave_MatlabTutorials/Assets/Iteration_3.png)
![alt text](/Week_8/Octave_MatlabTutorials/Assets/Iteration_4.png)
![alt text](/Week_8/Octave_MatlabTutorials/Assets/Iteration_10.png)

Load an image:

	A = double(imread('bird_small.png'));
or
	load ('bird_small.mat');
	A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

Size of the image:

	img_size = size(A);

Reshape the image into an Nx3 matrix where N = number of pixels.
Each row will contain the Red, Green and Blue pixel values
This gives us our dataset matrix X that we will use K-Means on.
	
	X = reshape(A, img_size(1) * img_size(2), 3);

Run your K-Means algorithm on this data

	K = 16; 
	max_iters = 10;

When using K-Means, it is important the initialize the centroids randomly: 

	initial_centroids = kMeansInitCentroids(X, K);

Run K-Means:

	[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

Find closest cluster members:

	idx = findClosestCentroids(X, centroids);

Essentially, now we have represented the image X as in terms of the indices in idx. 
We can now recover the image from the indices (idx) by mapping each pixel (specified by its index in idx) to the centroid value:

	X_recovered = centroids(idx,:);

Reshape the recovered image into proper dimensions:

	X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

Display the original image:

	subplot(1, 2, 1);
	imagesc(A); 
	title('Original');

Display compressed image side by side;
	
	subplot(1, 2, 2);
	imagesc(X_recovered)
	title(sprintf('Compressed, with %d colors.', K));

![alt text](/Week_8/Octave_MatlabTutorials/Assets/ImageCompression.png)

### Functions

__Find Closest Centroids__

	function idx = findClosestCentroids(X, centroids)

		K = size(centroids, 1);

		idx = zeros(size(X,1), 1);

		for i=1:size(X, 1)
    		distances = zeros(size(centroids, 1), 1);
    		for k=1:K
        		distances(k) = sumsq(X(i,:) - centroids(k, :));
    		end;
    		[minDistance, minIndex] = min(distances);
    		idx(i) = minIndex;
		end;
	end

__Compute Centroids__

	function centroids = computeCentroids(X, idx, K)

		[m n] = size(X);
		centroids = zeros(K, n);

		for k=1:K
    		% use logical arrays for indexing
    		% see http://www.mathworks.com/help/matlab/math/matrix-indexing.html#bq7egb6-1
    		indexes = idx == k;
    		centroids(k, :) = mean(X(indexes, :));
		end;
	end

__kMeans Init Centroids__

	function centroids = kMeansInitCentroids(X, K)

		centroids = zeros(K, size(X, 2));
		randidx = randperm(size(X, 1));
		centroids = X(randidx(1:K), :);
	end

__Run K-means__

	function [centroids, idx] = runkMeans(X, initial_centroids, max_iters, plot_progress)

		% Set default value for plot progress
		if ~exist('plot_progress', 'var') || isempty(plot_progress)
    		plot_progress = false;
		end

		% Plot the data if we are plotting progress
		if plot_progress
    		figure;
    		hold on;
		end

		% Initialize values
		[m n] = size(X);
		K = size(initial_centroids, 1);
		centroids = initial_centroids;
		previous_centroids = centroids;
		idx = zeros(m, 1);

		% Run K-Means
		for i=1:max_iters    
    		% Output progress
    		fprintf('K-Means iteration %d/%d...\n', i, max_iters);
    		if exist('OCTAVE_VERSION')
        		fflush(stdout);
    		end
    
    		% For each example in X, assign it to the closest centroid
    		idx = findClosestCentroids(X, centroids);
    
    		% Optionally, plot progress here
    		if plot_progress
        		plotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
        		previous_centroids = centroids;
        		fprintf('Press enter to continue.\n');
        		pause;
    		end
    
    		% Given the memberships, compute new centroids
    		centroids = computeCentroids(X, idx, K);
		end

		% Hold off if we are plotting progress
		if plot_progress
    		hold off;
		end
	end
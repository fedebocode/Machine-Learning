# Octave/Matlab Tutorial

List of common Octave commands to operate with Matrices, Vectors and/or data in general.

__General utilities:__

	pwd 						// Assign local directory to a a variable ans
	PS1(">> ") 					// Remove "octave" from terminal commands

	who							// Display stored variables in the current session
	whos						// Display all details of the above
	clear 						// Delete all store variables
	clear(M)					// Delete specific variable	

__I/O utilities:__
	
	load file.dat 				// Load file
	load ('file.dat')			// Load file
	save file.dat 				// Save file
	save file.txt v -ascii		// Save file as text (ASCII)

__Matrices and Vectors:__

	M = eye(2); 				// Create an identity 2x2 matrix
	M = rand(2,2) 				// Create a 2x2 matrix with random values
	M_trans = transpose(M)/M'	// Transpose a matrix
	M_inv = pinv(M) or inv(M) 	// Invert a matrix (pinv is safer, it inverts always)
	M = [1,2 ; 3,4] 			// Create a 2x2 matrix with values
	M = [1 ; 2 ; 3] 			// Create a 3x1 matrix 
	M = (:,2) 					// Get second column entries
	M = (:,1) 					// Get first column entries
	M = (1,:) 					// Get first row entries
	M = (2,:) 					// Get second row entries
	M(1,:) = [2 , 3]			// Assign values to row
	M(:,1) = [2 , 3]			// Assign values to column
	M(3,2)						// Return value at entry (3,2)
	M([1 3],:)					// Return values at row 1 and row 3
	M(:,[1 3])					// Return values at column 1 and column 3 
	M = [M, [10;20]]			// Append another column vector to the right
	M(:)						// Put all elements of M into a single vector
	size(M)						// Return longest dimension of M
	M = [A B]					// Concatenate two matrices A and B along the rows
	M = [A;B]					// Concatenate two matrices A and B along the columns
	M = A .*B 					// Multiply element wise
	M .^2						// Element wise squaring
	log(M)						// Element wise logarithm
	abs(M)						// Element wise absolute value
	find(M < 3)					// Return values less then 3
	sum(M)						// Sum column wise
	sum(M,1)					// Sum column wise
	sum(sum(M))					// Sum all entries of the matrix
	prod(M)						// Product column wise
	floor(M)					// Floor matrix
	ceil(M)						// Ceil matrix	

__Plot:__

	plot(x) 					// Open the plot window and plot values
	xlabel('time')				// Assign "time" name to xAxis in the chart
	ylabel('time')				// Assign "time" name to yAxis in the chart
	legend('sin','cos')			// Assign "sin" and "cos" names to legend in the chart
	title('my plot')			// Assign "my plot" name as title in the chart
	print -dpng 'myplot.png'	// Export chart as png
	close						// Close chart
	figure(1) = plot(M)			// Plot chart in a separate window
	figure(2) = plot(N)			// Plot chart in a separate window
	subplot(1,2,1)				// Plot a portion of the chart between 1 and 2
	imagesc(M)					// Plot a matrix
	colorbar					// Add color bar legend on the side
	colormap gray				// Change color chart map

__Loops__:

Syntax for a "for" loop:

	for i = 1:10
		i = i + 1
		disp(i)
	end

Syntax for a "while" loop:

	i = 1
	while true
		v(i) = 0;
		i = i + 1
		if i == 6
			break
		end
	end

__Advanced__:

	addpath('C:\Users\Federico\Documents')		// Add a search path for Octave to refer


Create a file for example called CostFunction.m in your directory which contains the following:

	function J = CostFunction(X,y,theta)

	% X is the "design matrix" containing our training examples
	% y is the class labels
	% theta is the values for which we want to solve teh cost function

	m = size(X,1);							% Number of training examples
	predictions = X * theta;				% Predictions of hypothesis on all m
	squaredErrors = (predictions - y).^2;	% Squared errors 
	
	J = 1/(2 * m) * sum(squaredErrors);

Now in Octave, define the variables and call the function:

	X = [1 1; 1 2; 1 3];
	y = [1; 2; 3]
	theta = [0; 1]
	
	J = CostFunction(X,y,theta)
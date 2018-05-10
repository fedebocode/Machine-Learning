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
	M_trans = transpose(M) 		// Transpose a matrix
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
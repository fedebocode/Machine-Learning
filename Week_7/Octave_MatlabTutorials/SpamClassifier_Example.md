# Spam Classifier Example

To use an SVM to classify emails into Spam v.s. Non-Spam, you first need to convert each email into a vector of features. In this part, you will implement the preprocessing steps for each email.

Initialization:

	clear ; close all; clc

You will convert each email into a vector of features in R^n:

	file_contents = readFile('emailSample1.txt');
	word_indices  = processEmail(file_contents);
	features      = emailFeatures(word_indices);

Load the Spam Email dataset:

	load('spamTrain.mat');
	C = 0.1;
	model = svmTrain(X, y, C, @linearKernel);
	p = svmPredict(model, X);

	fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

After training the classifier, we can evaluate it on a test set. We have included a test set in spamTest.mat

Load the test dataset:

	load('spamTest.mat');
	p = svmPredict(model, Xtest);

	fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);

Since the model we are training is a linear SVM, we can inspect the weights learned by the model to understand better how it is determining whether an email is spam or not. The following code finds the words with the highest weights in the classifier. Informally, the classifier 'thinks' that these words are the most likely indicators of spam.

Sort the weights and obtain the vocabulary list:

	[weight, idx] = sort(model.w, 'descend');
	vocabList = getVocabList();

	for i = 1:15
    	fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
	end

Now that you've trained the spam classifier, you can use it on your own emails! In the starter code, we have included spamSample1.txt, spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 

Set the file to be read in (change this to spamSample2.txt, emailSample1.txt or emailSample2.txt to see different predictions on different emails types). Try your own emails as well!

	filename = 'spamSample1.txt';

Read and predict:

	file_contents = readFile(filename);
	word_indices  = processEmail(file_contents);
	x = emailFeatures(word_indices);
	p = svmPredict(model, x);

	fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
	fprintf('(1 indicates spam, 0 indicates not spam)\n\n');
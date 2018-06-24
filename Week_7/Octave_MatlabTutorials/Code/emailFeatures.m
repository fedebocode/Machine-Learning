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

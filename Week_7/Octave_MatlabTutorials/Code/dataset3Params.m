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

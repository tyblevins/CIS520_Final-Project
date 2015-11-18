function [score_train,score_test,numpc] = pca_getpc(train_x,test_x)

%   input: original X for training and testing
%   output: PCAed X for training and testing, number of PCs that you
%   selected
[coeff_train, latent] = pcacov(train_x'*train_x);
score_train = train_x * coeff_train;
score_test = double(test_x) * coeff_train;
%figure, plot(cumsum(latent)/sum(latent));

% Set you numpc here
numpc = 10;
end


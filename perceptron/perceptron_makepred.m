function yhat = perceptron_makepred(X, w)
%Simple function for efficiently computing training error for perceptron
% X is padded Nx(D+1) feature set, Y is Nx1 training labels, w is (D+1)x1
% weights including bias term.
%
% Usage: 
%       error = perceptron_error(X, Y, w)
% Returns a scalar representing the training error (from 0 to 1, 0 being no error)

yhat = sign(X*w);
yhat(yhat == -1) = 0;  %format data as given
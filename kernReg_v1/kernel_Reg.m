function Yhat = kernel_Reg(X, X2,Ytrain)
% Evaluates the Histogram Intersection Kernel
%
% Usage:
%
%    Yhat = kernel_Reg(X, X2,Ytrain)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the histogram
% intersection kernel.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);
Yhat = zeros(m, 1);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% polynamial kernel based on choic of p
% p = 1;

% HINT: This should be a very straightforward one liner!
% K = (X2*X').^p;
% 
% % After you've computed K, make sure not a sparse matrix anymore
% K = full(K);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% gaussian kernel

% YOUR CODE GOES HERE.
sigma = 1;
Xtra = X';
X2tra = X2';


for i = 1:size(X2tra,2);
  K(i,:) = exp(-rdivide(full(sum(bsxfun(@minus, Xtra, X2tra(:,i)).^2,1)),(2*sigma^2)));
  %gaussian kernel
%     K(i,:) = full(X2(i,:) * X');
    Yhat(i) = sum(Ytrain'.*K(i,:),2)./sum(K(i,:),2); 
    
end
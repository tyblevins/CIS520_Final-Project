function Yhat = kernel_intersectionReg(X, X2,Ytrain)
% Evaluates the Histogram Intersection Kernel
%
% Usage:
%
%    K = KERNEL_INTERSECTION(X, X2, Ytrain)
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

% YOUR CODE GOES HERE.
sigma = 1;

for i = 1:size(X2,1);
%     K(i,:) = full(sum(bsxfun(@min, X, X2(i,:)),2))';   %intersection kernel
    
    K(i,:) = exp(-rdivide(full(sum(bsxfun(@minus, X, X2(i,:)).^2,2)),(2*sigma^2)));  %gaussian kernel
    
    Yhat(i) = sum(Ytrain'.*K(i,:),2)./sum(K(i,:),2); 
    
end
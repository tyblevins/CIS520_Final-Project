function [averaged_w err] = averaged_perceptron_train(X, Y, update_step_fnc, numPasses)
% Trains an averaged perceptron on a sparse set of examples (X, Y)
%
% Example Usage:
%    update_constant_0_5 = @(x,y,w) update_constant(x,y,w,0.5)
%    [model err] = averaged_perceptron_train(Xtrain, Ytrain, @update_constant_0_5, 2)
%
% For a N x D sparse feature matrix X and Nx1 label matrix Y, returns
% averaged D x 1 weight model
% 
% For each example x_i, the weight vector will be updated by 
%           w = w + update_step_fnc(x_i,y_i,w)*x_i;
%
% numPasses is the number of times of passing the whole dataset through perceptron.
% The loop will end after number of passes (numPasses) is reached.
% 
% The function should also return an (NumPasses*N)x1 vector err containing the training
% error of each averaged weight vector. 
% 

%For your convenience 
[sizeX,D] = size(X);

N = 500; %minbatch
%Initialize weights
w = 0.0001*ones(D,1);
%Keep a separate running sum of weights from each iteration
averaged_w = zeros(D,1);

err = zeros(numPasses*N,1);

%%YOUR CODE GOES HERE
for pass = 1:numPasses
    
    batch = randperm(sizeX,N);
    
    X_batch = X(batch,:);
    Y_batch = Y(batch,:);
    
    for i = 1:N
        w = w + update_step_fnc(X_batch(i,:),Y_batch(i),w)*X_batch(i,:)';
        
        averaged_w = mean(w + averaged_w,2);
%         averaged_w = w./2;
        err((pass-1)*N + i) = perceptron_error(X, Y, averaged_w);
%         disp(['Error @ Step ' num2str(i) ' = ' err(i*numPasses)])
    end
end
averaged_w = averaged_w./(numPasses*N);
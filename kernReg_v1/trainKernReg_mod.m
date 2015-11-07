%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% TRAIN RREGRESSION MODEL
% NOTE: This is a super simple start and results in RMSE_train = 0.511
% We need to do some sort of regularization for sure on this.

clear;


%load data
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

%also load dictionaries for visualization
load ../data/word_dictionary.mat
load ../data/bigram_dictionary.mat

X =[city_train word_train bigram_train];
% X =[city_train bigram_train];
Y = price_train;
X_test = [city_test word_test bigram_test];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numCities = size(city_train,2);
% sort by city
%get N random observations from each city (or select the ones that explain
%the most variance????)
N = 200;
X_red = zeros(N*numCities,size(X,2));
Y_red = zeros(N*numCities,1);

for i = 1:numCities
   citIdx = zeros(size(city_train));
   citIdx(:,i) = ones(size(city_train,1),1);
   
   tmpCity = full(city_train(logical(citIdx)));
   
   tmpX = X(logical(tmpCity),:);
   tmpY = Y(logical(tmpCity),:);
   
   newIdx = randperm(size(tmpX,1));
   newIdx = newIdx(1:N);
   
   %assign the random new X's to the reduced matrix
   X_red(((i-1)*N+1):(i*N),:) = full(tmpX(newIdx,:));
   Y_red(((i-1)*N+1):(i*N),:) = full(tmpY(newIdx,:));
end





%create a new train X vector to compare against all the observations that were 
%not used in new train X

tic;
%
%% calculate training error
disp('calculating Kernel matrix')
    Yhat_train = kernel_Reg(sparse(X_red), X, (Y_red));
toc;
% [Yhat_train] = full(kernreg_test(4, X, Y, X));
% toc;


RMSE = sqrt(((1/length(Y))*sum((Y-Yhat_train).^2)))



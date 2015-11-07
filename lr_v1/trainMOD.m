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


tic;
disp('training model...')
w = mldivide((X'*X),(X'*Y));
toc

%
%% calculate training error
tic;
disp('calculating training error...')
Yhat_train = full(X*w);
toc;


RMSE = sqrt(((1/length(Y))*sum((Y-Yhat_train).^2)))



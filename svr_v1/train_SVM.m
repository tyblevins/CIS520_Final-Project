%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% TRAIN KERNEL SVM
% NOTE:
% This script should be run from folder svm_v1

clear;
addpath ../tools/libsvm


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

% X =[city_train word_train bigram_train];
X =[city_train bigram_train];
Y = price_train;

X_test = [city_test word_test bigram_test];

tic;
disp('training model...')
model = svmtrain(Y, X, '-s 3 -t 0 -c 1 -p 0.1');
toc

%
%% calculate training error
disp('calculating training error...')
tic;
[yHat_train, trainAcc, dec_values] = svmpredict(Y, X, model); 
toc;


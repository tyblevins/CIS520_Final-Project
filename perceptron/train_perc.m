%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% Perceptron
% NOTE:
% This script should be run from folder svm_v1

clear;

%% Load in data
addpath(genpath('..\CIS520_twitter_data'))
addpath(genpath('perceptron'))
addpath(genpath('CIS520_Final-Project'))

disp(sprintf('loading in data........\n'))
gender_train = dlmread('genders_train.txt');
% image_raw   = dlmread('images_train.txt');
% image_feats = dlmread('image_features_train.txt');
words_train = dlmread('words_train.txt');

%load dictionary
[rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');

%find stop words
%predefined
% stop_words  = textread('default_stopVoc.txt','%s','delimiter','\n');
%most frequent
[wordfreq, freqidx] = sort(sum(words_train,1),'descend');
stopRemove = 50;
voc{freqidx(1:stopRemove)}   %show the 10 words which have largest effect on 1st pca
disp(sprintf('Done Loading Data (stop words above)! \n'))




%remove top 50
words_train(:,freqidx(1:stopRemove)) = [];

%assign feats
Y = gender_train;
X = [words_train];

%% train stuff
%define x and y's from data above
numTr = ceil(size(Y,1)*.70);

trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

%define train set
Y_train = Y(trainIdx,:);
X_train = X(trainIdx,:);

%defin evaluation set
Y_eval = Y(evalIdx,:);
X_eval = X(evalIdx,:);

% INSTRUCTIONS: Use the averaged_perceptron_train function to train model
% using learning rate of 1.0
numPasses = 8; %Do not change

%choose update func
update_fnc = @(x,y,w) update_passive_aggressive(x,y,w);
% update_fnc = @(x,y,w) update_constant(x,y,w,1.0);

%run perceptron algorithm
[w_avg err] = averaged_perceptron_train_miniBatch(X_train, Y_train, update_fnc, numPasses);
results.w_const  = w_avg;%Averaged W for constant learning rate 1.0
results.train_err_const = err;%Error vector for constant learning rate 1.0
results.test_err_const = perceptron_error(X_eval,Y_eval, w_avg);
disp(['final averaged error:  ', num2str(results.test_err_const)]);



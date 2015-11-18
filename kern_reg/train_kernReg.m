%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% TRAIN KERNEL SVM
% NOTE:
% This script should be run from folder svm_v1

clear;

%% Load in data
addpath(genpath('CIS520_twitter_data'))
addpath(genpath('svm'))
addpath(genpath('CIS520_Final-Project'))

disp(sprintf('loading in data........\n'))
gender_train = dlmread('genders_train.txt');
% image_raw   = dlmread('images_train.txt');
% image_feats = dlmread('image_features_train.txt');
words_train = dlmread('words_train.txt');
disp(sprintf('Complete! \n'))

%load dictionary
[rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');

%find stop words
[wordfreq, freqidx] = sort(sum(words_train,1),'descend');
stopRemove = 40;

voc{freqidx(1:stopRemove)}   %show the 10 words which have largest effect on 1st pca

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


%% transform features
% [coeff,score,latent,tsquared,explained,mu] = pca(X_train,'NumComponents',50);
[coeff,score,latent] = pca(X_train,'NumComponents',500);


%look at the coef 
[bigvalues, bigidx] = sort(coeff(:,1), 'descend');
voc{bigidx(1:10)}   %show the 10 words which have largest effect on 1st pca

explainedPC = cumsum(latent)./sum(latent);

%find 90% explained
%minimum PC to explain 90% variance
numpc =  find((explainedPC > .90),1);

PC_train = X_train * coeff;
PC_eval =  X_eval * coeff;

PC_train = PC_train(:, 1:numpc);
PC_eval =  PC_eval(:, 1:numpc);


%%%%%%%%%%%%%%%%%%%%%%%kernels
kern = @(x,x2) kernel_poly(x, x2, 1);

tic;
disp('training model...')
yhat = kernreg_test(1, PC_train, Y_train, PC_eval);
toc;



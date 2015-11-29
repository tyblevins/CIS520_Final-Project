%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% TRAIN DEC TREE
% NOTE:
% This script should be run from folder svm_v1

clear;

%% Load in data
addpath(genpath('..\CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('featSel'))

disp(sprintf('loading in data........\n'))


%% TRAIN DATA
gender_train = dlmread('genders_train.txt');
% image_raw_train   = dlmread('images_train.txt');
image_feats_train = dlmread('image_features_train.txt');
words_train = dlmread('words_train.txt');


%Load L0 featsel results
load('knn_vocSel.mat')


%% load dictionary and find stop words
[rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');

%find stop words
%predefined dictionary
% stop_words  = textread('default_stopVoc.txt','%s','delimiter','\n');
% load('knn_vocSel.mat')
% rmvStop = find(sum(cell2mat(cellfun(@strcmp,repmat(voc,1,length(stop_words))' , repmat(stop_words,1,size(voc,1)),'UniformOutput',0)),1)' == 1);
% featSel_resultsKNN(rmvStop) = 0;  %add stop words to feature selection results

% voc{rmvStop}   
% words_train(:,featSel_resultsKNN == 0) = [];
%find words that are only used by 10 or less users and remove them.
% notUsed_Idx = (sum(words_train ~= 0,1) <= 10);
% words_train(:,notUsed_Idx) = [];


%% TEST DATA
% image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');

% words_test(:,featSel_resultsKNN == 0) = [];
% words_test(:,notUsed_Idx) = [];

disp(sprintf('Done Loading Data \n'))


%assign feats
Y = gender_train;
Y(Y ==0) = -1;

X = [words_train image_feats_train];
X_test = [words_test image_feats_test];

%normalize
X = X + 2e-13;
avgX = mean(X,1);
stdX = std(X)+2e-13;
X = bsxfun(@rdivide ,((X)  - repmat(avgX,size(X,1),1)), stdX);
X_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);


disp(sprintf('Training \n'))
numFolds = 5;
numlambda = 20;
lambdaTest = logspace(-1,-2,100);
alpha = 0.5;
%featsel using lasso
tic;
opts = statset('UseParallel',true);  %Do dis shit in parallel!!!  use number of availble workers for CV
% [fLasso, tINFO] = lasso(X, Y,'alpha',alpha,'CV',numFolds,'NumLambda',numlambda,'Options',opts);[fLasso, tINFO] = lasso(X, Y,'alpha',alpha,'CV',numFolds,'NumLambda',numlambda,'Options',opts);
[fLasso, tINFO] = lasso(X, Y,'alpha',alpha,'CV',numFolds,'lambda',lambdaTest,'Options',opts);

% [fLasso, tINFO] = lassoglm(X, Y,'binomial','alpha',alpha,'CV',numFolds,'NumLambda',numlambda,'Options',opts);

solTime = toc

figure(1)
lassoPlot(fLasso,tINFO,'plottype','CV');


% bestIdx = tINFO.Index1SE;
% lambda  = tINFO.Lambda1SE;
bestIdx = tINFO.IndexMinMSE;
lambda  = tINFO.LambdaMinMSE;

w   = fLasso(:,bestIdx);
int = tINFO.Intercept(bestIdx);
numFeats_w = sum(w ~= 0);

%show words that lasso picked
voc{w(1:5000) ~= 0}


yhat_train = sign(X*w + int); 

train_acc = mean(Y == yhat_train);


lassoResults = {w, int, numFeats_w};
save('lassoFeatSel.mat','lassoResults');




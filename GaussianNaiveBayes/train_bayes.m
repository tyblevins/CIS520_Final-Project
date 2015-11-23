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
addpath(genpath('GaussianNaiveBayes'))

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
stop_words  = textread('default_stopVoc.txt','%s','delimiter','\n');
load('knn_vocSel.mat')
rmvStop = find(sum(cell2mat(cellfun(@strcmp,repmat(voc,1,length(stop_words))' , repmat(stop_words,1,size(voc,1)),'UniformOutput',0)),1)' == 1);
featSel_resultsKNN(rmvStop) = 0;  %add stop words to feature selection results
% 
% % voc{rmvStop}   
words_train(:,featSel_resultsKNN == 0) = [];
% %find words that are only used by 10 or less users and remove them.
notUsed_Idx = (sum(words_train ~= 0,1) <= 10);
words_train(:,notUsed_Idx) = [];


%% TEST DATA
% image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');

words_test(:,featSel_resultsKNN == 0) = [];
words_test(:,notUsed_Idx) = [];

disp(sprintf('Done Loading Data \n'))


%assign feats
Y = gender_train;
X = [words_train image_feats_train];
X_test = [words_test image_feats_test];
% X = [words_train ];
% X_test = [words_test ];
%mean center data
% X = X + 2e-13;
% avgX = mean(X,1);
% stdX = std(X)+2e-13;
% X = bsxfun(@rdivide ,((X)  - repmat(avgX,size(X,1),1)), stdX);
% X_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);

%mean center only
% X = ((X)  - repmat(avgX,size(X,1),1));
% X_test = ((X_test)  - repmat(avgX,size(X_test,1),1));

% X = X + 0.01.*normrnd(0,1,size(X));  %add noise

%% train stuff
%define x and y's from data above
numTr = ceil(size(Y,1)*.70);

trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

%define train set
Y_train = Y(trainIdx,:);
X_train = X(trainIdx,:) ;

%defin evaluation set
Y_eval = Y(evalIdx,:);
X_eval = X(evalIdx,:);
%

%evaluate dat shit
yhat = GaussianNaiveBayes(X_train,Y_train,X_eval);
acc_bayes = mean(Y_eval == yhat);

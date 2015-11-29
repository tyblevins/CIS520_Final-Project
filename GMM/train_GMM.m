%Jared Wilson 
%GMM TEST
%11/3/2015
% CIS520 -- Final Project Fall 2015
clear;

%% Load in data
addpath(genpath('..\CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('GMM'))

disp(sprintf('loading in data........\n'))


%% TRAIN DATA
gender_train = dlmread('genders_train.txt');
% image_raw_train   = dlmread('images_train.txt');
image_feats_train = dlmread('image_features_train.txt');
words_train = dlmread('words_train.txt');

%% load dictionary and find stop words
[rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');

%find stop words
%predefined dictionary
% stop_words  = textread('default_stopVoc.txt','%s','delimiter','\n');
% rmvStop = find(sum(cell2mat(cellfun(@strcmp,repmat(voc,1,length(stop_words))' , repmat(stop_words,1,size(voc,1)),'UniformOutput',0)),1)' == 1);
% voc{rmvStop}   %show the 10 words which have largest effect on 1st pca
% words_train(:,rmvStop) = [];
% notUsed_Idx = (sum(words_train,1) <= 10);
% words_train(:,notUsed_Idx) = [];



%% TEST DATA
% image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');
% 
% words_test(:,rmvStop) = [];
% words_test(:,notUsed_Idx) = [];

disp(sprintf('Done Loading Data \n'))


Y = gender_train;
Y(Y ==0) = -1;

X = [words_train image_feats_train];
X_test = [words_test image_feats_test];

load('lassoFeatSel.mat')
w = lassoResults{1};
lassoRmv = (w == 0);
X(:,lassoRmv) = [];
X_test(:,lassoRmv) = [];

X = X + 2e-13;
avgX = mean(X,1);
stdX = std(X)+2e-13;
X = bsxfun(@rdivide ,((X)  - repmat(avgX,size(X,1),1)), stdX);
X_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);


numTr = ceil(size(Y,1)*.70);

trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

%define train set
Y_train = Y(trainIdx,:);
X_train = X(trainIdx,:) ;

%defin evaluation set
Y_eval = Y(evalIdx,:);
X_eval = X(evalIdx,:);

k = 4;
tic
% GMModel = fitgmdist(X_train,k,'CovarianceType','diagonal','SharedCovariance',true);
GMModel = fitgmdist(X_train,k,'CovarianceType','diagonal','SharedCovariance',true,'Replicates',10);

toc

idx = cluster(GMModel,X_train);

cluster_Lab = zeros(k,1);
yhat        = zeros(length(idx),1)
%take majority vote in each cluster
for i = 1:k
    cluster_Lab(i) = sign(mean(Y(idx == i)));
    yhat(idx == i) = cluster_Lab(i);
end

train_acc = mean(Y_train == yhat);










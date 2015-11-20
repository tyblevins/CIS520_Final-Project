%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% Perceptron
% NOTE:
% This script should be run from folder svm_v1

clear;

%% Load in data
addpath(genpath('..\CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('perceptron'))

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
stop_words  = textread('default_stopVoc.txt','%s','delimiter','\n');
rmvStop = find(sum(cell2mat(cellfun(@strcmp,repmat(voc,1,length(stop_words))' , repmat(stop_words,1,size(voc,1)),'UniformOutput',0)),1)' == 1);
voc{rmvStop}   %show the 10 words which have largest effect on 1st pca
words_train(:,rmvStop) = [];
notUsed_Idx = (sum(words_train ~= 0,1) <= 10);
words_train(:,notUsed_Idx) = [];

%most frequent
% [wordfreq, freqidx] = sort(sum(words_train,1),'descend');
% stopRemove = 25;
% rmvStop = freqidx(1:stopRemove);
% voc{rmvStop}   %show the 10 words which have largest effect on 1st pca
% words_train(:,rmvStop) = [];
% notUsed_Idx = (sum(words_train,1) <= 10);
% words_train(:,notUsed_Idx) = [];

%% TEST DATA
% image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');


words_test(:,rmvStop) = [];
words_test(:,notUsed_Idx) = [];

disp(sprintf('Done Loading Data \n'))

%% FEATURE MANIPULATION STUFF
% load('nnmfW_100.mat');
% load('nnmfH_100.mat');
% 
% nnmf_feats_train = image_raw_train * H(1:10,:)';
% nnmf_feats_test  = image_raw_test  * H(1:10,:)';

    %KERNELS
% K = kernel_poly(words_train, words_train,1);
% Kgaus = kernel_gaussian(words_train, words_train,20);
% K2 = kernel_poly(words_train, words_train,2);
% K_test = kernel_poly(words_train, words_test,1);
% K2_test = kernel_poly(words_train, words_test,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%assign feats
Y = gender_train;
% X = [words_train image_feats_train K];
% X_test = [words_test image_feats_test K_test];
% 
X = [words_train image_feats_train ];
X_test = [words_test image_feats_test ];

%% standardize data
X = X + 2e-13;
avgX = mean(X,1);
% stdX = std(X)+2e-13;
% 
% X = bsxfun(@rdivide ,((X)  - repmat(avgX,size(X,1),1)), stdX);
% X_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);
X = (X)  -  repmat(avgX,size(X,1),1);
X_test = ((X_test)  - repmat(avgX,size(X_test,1),1));
%% train stuff
%define x and y's from data above
numTr = ceil(size(Y,1)*.70);

trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

%define train set
Y(Y == 0) = -1;  %change labels to -1/1 for peceptron learning

Y_train = Y(trainIdx,:);
X_train = X(trainIdx,:);

%defin evaluation set
Y_eval = Y(evalIdx,:);
X_eval = X(evalIdx,:);


%% DIM RED WITH PCA
% [coeff,score,latent] = pca(X,'NumComponents',500);
% 
% 
% %look at the coef 
% [bigvalues, bigidx] = sort(abs(coeff(:,4)), 'descend');
% voc{bigidx(1:10)}   %show the 10 words which have largest effect on 1st pca
% 
% explainedPC = cumsum(latent)./sum(latent);
% 
% percExp = 0.9;
% numpc =  find((explainedPC > percExp),1);
% 
% 
% PC_train = [ones(size(X_train, 1),1) X_train*coeff];
% PC_eval =  [ones(size(X_eval, 1),1) X_eval*coeff];
% 
% PC_train = PC_train(:, 1:numpc);
% PC_eval =  PC_eval(:, 1:numpc);

% INSTRUCTIONS: Use the averaged_perceptron_train function to train model
% using learning rate of 1.0
numPasses = 12; %Do not change

%choose update func
update_fnc = @(x,y,w) update_passive_aggressive(x,y,w);
% update_fnc = @(x,y,w) update_constant(x,y,w,1.0);

%% RUN EVALUATION
%run perceptron algorithm
disp(sprintf('TRAINING... \n'))
[w_avg err] = averaged_perceptron_train(X_train, Y_train, update_fnc, numPasses);
results.w_const  = w_avg; %Averaged W for constant learning rate 1.0
results.train_err_const = err; %Error vector for constant learning rate 1.0
results.test_err_const = perceptron_error(X_eval,Y_eval, w_avg);
disp(['final averaged error:  ', num2str(results.test_err_const)]);

%% TRAIN  MODEL
[w_avg err] = averaged_perceptron_train(X, Y, update_fnc, numPasses);
results.w_const  = w_avg;%Averaged W for constant learning rate 1.0
results.train_err_const = err;%Error vector for constant learning rate 1.0

yhat = perceptron_makepred(X_test, w_avg);

%save model and prediction
% save('w_percep.mat','w_avg');
% save('yhat_perc.mat','yhat');
% 
% 
% dlmwrite('submit.txt', yhat);



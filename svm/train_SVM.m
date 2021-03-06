%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% TRAIN KERNEL SVM
% NOTE:
% This script should be run from folder svm_v1
clear;

%% Load in data
addpath(genpath('..\CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('svm'))
addpath(genpath('tools'))

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

% voc{rmvStop}   
words_train(:,featSel_resultsKNN == 0) = [];
%find words that are only used by 10 or less users and remove them.
% words_train(:,rmvStop) = [];
notUsed_Idx = (sum(words_train ~= 0,1) <= 10);
words_train(:,notUsed_Idx) = [];


%% TEST DATA
% image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');

words_test(:,featSel_resultsKNN == 0) = [];
% words_test(:,rmvStop) = [];
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
X = [words_train image_feats_train];
X_test = [words_test image_feats_test];

%% standardize data
X = X + 2e-13;
avgX = mean(X,1);
stdX = std(X)+2e-13;
% 
X = bsxfun(@rdivide ,((X)  - repmat(avgX,size(X,1),1)), stdX);
X_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);

% X = (X)  -  repmat(avgX,size(X,1),1);
% X_test = ((X_test)  - repmat(avgX,size(X_test,1),1));
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

%%%%%%%%%%%%%%%%%%%%%%%%% SVM CROSS VALIDATION
%# grid of parameters
folds = 5;
[C,gamma] = meshgrid(2:2:6,-15:1:-10);

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
parfor i=1:numel(C)
    cv_acc(i) = svmtrain(Y_train, X_train, ...
                    sprintf('-t 2 -c %f -g %f -e 0.001 -v %d', 2^C(i), 2^gamma(i), folds));
end

%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);

%# contour plot of paramter selection
contour(C, gamma, reshape(cv_acc,size(C))), colorbar
hold on
plot(C(idx), gamma(idx), 'rx')
text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
    'HorizontalAlign','left', 'VerticalAlign','top')
hold off
xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')


tic
cv_accTest =  svmtrain(Y_train, X_train, sprintf('-t 2 -c 16 -e 0.001 -g 0.000030518 -h 1 -v 5'));
toc;


% Train and evaluate SVM classifier using libsvm
model = svmtrain(Y_train, X_train, sprintf('-t 2 -c 16 -e 0.001 -g 0.000030518'));
% model = svmtrain(Y_train, X_train, sprintf('-t 2 -c 1 -g 0.00005'));
% model = svmtrain(Y_train, X_train, sprintf('-t 1'));
[yhat acc vals] = svmpredict(Y_eval, X_eval, model);
test_acc = mean(yhat==Y_eval)


%train the full model
model = svmtrain(Y, X, sprintf('-t 2 -c 16 -e 0.001 -g 0.000030518'));

[yhat acc vals] = svmpredict(ones(size(X_test,1),1), X_test, model);
[yhat_train train_acc vals] = svmpredict(Y, X, model);


%save model and prediction
save('svm_mod.mat','model');
save('yhat_svm.mat','yhat');
save('yhat_svmTr.mat','yhat_train');


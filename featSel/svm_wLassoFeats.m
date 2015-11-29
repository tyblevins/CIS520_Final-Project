%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% test using lassoReg feats only
% NOTE:
% This script should be run from folder svm_v1

clear;

%% Load in data
addpath(genpath('..\CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('tools'))
addpath(genpath('featSel'))

disp(sprintf('loading in data........\n'))


%% TRAIN DATA
gender_train = dlmread('genders_train.txt');
% image_raw_train   = dlmread('images_train.txt');
image_feats_train = dlmread('image_features_train.txt');
words_train = dlmread('words_train.txt');

%% load dictionary and find stop words
[rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');

%% TEST DATA
% image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');

disp(sprintf('Done Loading Data \n'))


%assign feats
Y = gender_train;
Y(Y == 0) = -1;
X = [words_train image_feats_train];
X_test = [words_test image_feats_test];

%find stop words
%predefined dictionary
load('lassoFeatSel.mat')
lassoRmv = (lassoResults{1} == 0);

X(:,lassoRmv) = [];
X_test(:,lassoRmv) = [];

%mean center data
% X = X + 2e-13;
% avgX = mean(X,1);
% % stdX = std(X)+2e-13;
% X = bsxfun(@rdivide ,((X)  - repmat(avgX,size(X,1),1)), stdX);
% X_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);

%%%%%%%%%%%%%%%%%%%%%%%%% SVM CROSS VALIDATION
%# grid of parameters
folds = 5;
[C,gamma] = meshgrid(6:1:12,-22:2:-10);

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
parfor i=1:numel(C)
    cv_acc(i) = svmtrain(Y, X, ...
                    sprintf('-t 2 -c %f -g %f -v %d', 2^C(i), 2^gamma(i), folds));
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
cv_accTest =  svmtrain(Y, X, sprintf('-t 2 -c 256 -g 0.00000095367 -h 1 -v 5'));
toc;

tic
mod =  svmtrain(Y, X, sprintf('-t 2 -c 256 -g 0.00000095367 -h 1'));
toc;

[yhat acc vals] = svmpredict(ones(size(X_test,1),1), X_test, mod);
[yhat_train train_acc vals] = svmpredict(Y, X, mod);


%save model and prediction
save('svmLasso_mod.mat','mod');
save('yhat_svmLasso.mat','yhat');
save('yhat_svmLassoTr.mat','yhat_train');


%%
% Random Forest Train




%% 
% Ada Boost Train
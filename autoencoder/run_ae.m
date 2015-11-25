clear
clc

addpath(genpath('..\CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('autoencoder'))
addpath(genpath('tools'))

% load in data
% image_raw_train = dlmread('images_train.txt');
% image_raw_test = dlmread('images_test.txt');

gender_train = dlmread('genders_train.txt');
words_train = dlmread('words_train.txt');
words_test = dlmread('words_test.txt');



Y = gender_train;
Y(Y == 0) = -1;

train_x = words_train./max(words_train(:));
test_x   = words_test./max(words_test(:)); 

%% auto encoder
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');

% train_x = image_raw_train/255;
% test_x  = image_raw_test/255;


[ dbn ] = rbm( train_x );
[ new_feat, new_feat_test ] = newFeature_rbm( dbn,train_x,test_x );

tic
cv_acc =  svmtrain(Y, new_feat, sprintf('-t 0 -c 1 -v 5'));
toc;


%% logistic 
addpath('./liblinear');
%[ precision_ori_log ] = logistic( X_train, Y_train,X_test, Y_test );
%[ precision_pca_log ] = logistic( score_train(:, 1:numpc), Y_train, score_test(:,1:numpc), Y_test );
[ precision_ae_log ] = logistic( new_feat, Y_train, new_feat_test, Y_test );

%% kmeans

K = [10, 25];
precision_ae_km = zeros(length(K), 1);
for i = 1 : length(K)
    k = K(i);
    precision_ae_km(i) = k_means(new_feat, Y_train, new_feat_test, Y_test, k);
end


precision_pc_km = zeros(length(K), 1);
for i = 1 : length(K)
    k = K(i);
    precision_pc_km(i) = k_means(score_train(:, 1:numpc), Y_train, score_test(:,1:numpc), Y_test, k);
end

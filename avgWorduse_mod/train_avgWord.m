addpath(genpath('..\CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('avgWorduse_mod'))
addpath(genpath('tools'))
addpath(genpath('logistic_reg'))

disp(sprintf('loading in data........\n'))


%% TRAIN DATA
gender_train = dlmread('genders_train.txt');
% image_raw_train   = dlmread('images_train.txt');
image_feats_train = dlmread('image_features_train.txt');
words_train = dlmread('words_train.txt');


%% TEST DATA
% image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');

disp(sprintf('Done Loading Data \n'))


%calculate above average use of each word for each user.
avgWordUse = mean([words_train ; words_test],1);

neverUsed = (avgWordUse == 0);

avgWordUse(:,neverUsed) = [];
words_train(:,neverUsed) = [];
words_test(:,neverUsed) = [];

abvAvg_train = bsxfun(@gt,words_train,repmat(avgWordUse,size(gender_train,1),1));

%caclulate probabilty of femal/male given each word
prob_F = sum(gender_train == 1)/length(gender_train);
prob_M = sum(gender_train == 0)/length(gender_train);

probWord = sum(abvAvg_train,1)/sum(sum(abvAvg_train));

probWord_F = sum(abvAvg_train(gender_train == 1,:),1)/sum(sum(abvAvg_train(gender_train == 1,:))+2e-13);
probWord_M = sum(abvAvg_train(gender_train == 0,:),1)/sum(sum(abvAvg_train(gender_train == 0,:))+2e-13);

probF_word = (probWord_F*prob_F)./probWord;
probM_word = (probWord_M*prob_M)./probWord;

[bigvalues, bigidx] = sort(probF_word, 'descend');
voc{bigidx(1:10)}   %show the 10 words which have largest effect on 1st pca

[bigvalues, bigidx] = sort(probM_word, 'descend');
voc{bigidx(1:10)}   %show the 10 words which have largest effect on 1st pca



Y(Y ==0) = -1;
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

tic;
% make a model and predictions
% acc = train(Y, sparse(X), ['-s 0 -c 0.0055 -B 1 -v 5', 'col']);
acc = train(Y, sparse(X), ['-s 0 -c 0.00097656 -B 1 -v 5', 'col']);
toc;



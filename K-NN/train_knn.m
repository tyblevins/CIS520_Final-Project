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
addpath(genpath('K-NN'))

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

%assign feats
Y = gender_train;
X = [words_train image_feats_train];
X_test = [words_test image_feats_test];

%mean center data
X = X + 2e-13;
avgX = mean(X,1);
stdX = std(X)+2e-13;

X = bsxfun(@rdivide ,((X)  - repmat(avgX,size(X,1),1)), stdX);
X_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);

%% train stuff
%define x and y's from data above
numTr = ceil(size(Y,1)*.70);

trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

%define train set
% Y(Y == 0) = -1;  %change labels to -1/1 for peceptron learning

Y_train = Y(trainIdx,:);
X_train = X(trainIdx,:);

%defin evaluation set
Y_eval = Y(evalIdx,:);
X_eval = X(evalIdx,:);


%% transform features
% [coeff,score,latent,tsquared,explained,mu] = pca(X_train,'NumComponents',50);
[coeff,score,latent] = pca([X; X_test]);


%look at the coef 
[bigvalues, bigidx] = sort(coeff(:,1), 'descend');
voc{bigidx(1:10)}   %show the 10 words which have largest effect on 1st pca

explainedPC = cumsum(latent)./sum(latent);



%% CV to find number of pc to use
% PC = X*coeff;
% numParam = 100;
% pc_CV = (linspace(0.5,0.8,numParam));  %10 different values
% numFolds = 10;
% cvIdx = crossvalind('Kfold', length(Y_train), numFolds);
% 
% cvAcc = zeros(numFolds,numParam);
% k = 13;  %value found through cv kinda
% disp(sprintf('starting CV........\n'))
% parfor expIdx = 1:numel(pc_CV)
%     for cv = 1:numFolds
%         numpc =  find((explainedPC > pc_CV(expIdx)),1);
%         tmpPC = PC(:, 1:numpc);
%         PC_fold     = tmpPC(cvIdx ~= cv,:);
%         Y_fold      = Y(cvIdx ~= cv);
%         PC_foldEval = tmpPC(cvIdx == cv,:);
%         Y_foldEval  = Y(cvIdx == cv);
%         tic;
%         mdl = fitcknn(PC_fold,Y_fold, 'NumNeighbors',k);     
%         toc;
%         yhat_knn = predict(mdl,PC_foldEval)        
%         
%         cvAcc(cv,expIdx) = sum(Y_foldEval == yhat_knn)/length(yhat_knn);
%         
%         disp(['Progress: Parameter - ' num2str(expIdx) '/' num2str(numParam) ...
%             ' Fold - ' num2str(cv) '/' num2str(numFolds) sprintf('\n')])
% 
%     end
% end
% 
% parAcc = mean(cvAcc,1);
% [~, bestIdx] = max(parAcc);
% 
% figure(1)
% plot(pc_CV,parAcc);
% 
% percExp = pc_CV(bestIdx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cv solution 53.03%  (ref code in train_decTree.m)
percExp = 0.5303;
numpc =  find((explainedPC > percExp),1);
% numpc =  200;

% PC_train = [ones(size(X_train, 1),1) X_train*coeff];
% PC_eval =  [ones(size(X_eval, 1),1) X_eval*coeff];
PC = X*coeff;
PC_train = [X_train*coeff];
PC_eval =  [X_eval*coeff];


PC_train = PC_train(:, 1:numpc);
PC_eval =  PC_eval(:, 1:numpc);
PC      =  PC(:, 1:numpc);


%% EVALUATION
k = 13;
mdl = fitcknn(PC_train,Y_train,'NumNeighbors', k);
yhat_knn = predict(mdl,PC_eval);

acc_Knn = sum(Y_eval == yhat_knn)/length(yhat_knn)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%make predictions

%make model with all data
mdl = fitcknn(PC,Y,'NumNeighbors', k);

PC_test =  [X_test*coeff];
PC_test = PC_test(:, 1:numpc);

yhat_knn = predict(mdl,PC_test);

%save model and prediction
save('knn_mod.mat','mdl');
save('yhat_knn.mat','yhat_knn');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find best k   should be around 15
% 
% k = 1:50;
% acc_Knn = zeros(size(k));
% for i = 1:50
%     
%     mdl = fitcknn(PC_train,Y_train,'NumNeighbors', k(i));
%     yhat_knn = predict(mdl,PC_eval);
%     acc_Knn(i) = sum(Y_eval == yhat_knn)/length(yhat_knn);
%     disp(num2str(i))
% %     yhat_knn = knnclassify(PC_eval, PC_train, Y_train, k(i));
% %     acc_Knn(i) = sum(Y_eval == yhat_knn)/length(yhat_knn);
% end
% 
% plot(k,acc_Knn);

% acc_Knn = sum(Y_eval == yhat_knn)/length(yhat_knn);




%%%%%%%%%%%%%%%%%%%kernels regression NOT SO GOOD AND SUPER COMPUTATIONALLY
%%%%%%%%%%%%%%%%%%%INTENSIVE
% tic;
% disp('training model...')
% yhat = kernreg_test(1, PC_train, Y_train, PC_eval,'linf');
% toc;
% yhat = sign(yhat);
% 
% accKern = sum(Y_eval == yhat)/length(yhat);




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
addpath(genpath('Random_Forest'))

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
% featSel_resultsKNN(rmvStop) = 0;  %add stop words to feature selection results

% voc{rmvStop}   
words_train(:,featSel_resultsKNN == 0) = [];
%find words that are only used by 10 or less users and remove them.
% notUsed_Idx = (sum(words_train ~= 0,1) <= 10);
% words_train(:,notUsed_Idx) = [];

%remove words that are used similarly between males and females
% I think this overfits
% maleCount   = sum(words_train(gender_train == 0,:),1);
% femaleCount = sum(words_train(gender_train == 1,:),1);
% totalWordUse = sum(words_train,1);
% 
% absDif_count = abs(maleCount - femaleCount);
% 
% remvSim_Idx = (absDif_count <= totalWordUse*0.05); %remove if difference is less than 5% of total word use
% words_train(:,remvSim_Idx) = [];

%most frequent
% [wordfreq, freqidx] = sort(sum(words_train,1),'descend');
% stopRemove = 25;
% rmvStop = freqidx(1:stopRemove);
% voc{rmvStop}   %show the 10 words which have largest effect on 1st pca
% words_train(:,rmvStop) = [];
% notUsed_Idx = (sum(words_train ~= 0,1) <= 10);
% words_train(:,notUsed_Idx) = [];

%% TEST DATA
% image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');

words_test(:,featSel_resultsKNN == 0) = [];
% words_test(:,notUsed_Idx) = [];

disp(sprintf('Done Loading Data \n'))


%assign feats
Y = gender_train;
X = [words_train image_feats_train];
X_test = [words_test image_feats_test];

%mean center data
% X = X + 2e-13;
% avgX = mean(X,1);
% % stdX = std(X)+2e-13;
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


%% transform features
% [coeff,score,latent,tsquared,explained,mu] = pca(X_train,'NumComponents',50);
% [coeff,score,latent] = pca([X; X_test]);  %use test and train for PCA
% % [coeff,score,latent] = pca([X],'NumComponents',1500);
% 
% 
% %look at the coe12f 
% [bigvalues, bigidx] = sort(abs(coeff(1:size(words_train,2),2)), 'descend');
% voc{bigidx(1:10)}   %show the 10 words which have largest effect on 1st pca
% 
% explainedPC = cumsum(latent)./sum(latent);


%% CV to find number of pc to use
% PC = [ones(size(X, 1),1) X*coeff];
% numParam = 10;
% pc_CV = (linspace(0.8,1,numParam));  %10 different values
% numFolds = 10;
% cvIdx = crossvalind('Kfold', length(Y_train), numFolds);
% 
% cvAcc = zeros(numFolds,numParam);
% 
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
%         mod = TreeBagger(250,PC_fold,Y_fold,'Method','classification','OOBPred','On','MinLeaf',1,'FBoot',0.5);
%         toc;
%         yhat = str2num(cell2mat(predict(mod,PC_foldEval)));
%         
%         
%         cvAcc(cv,expIdx) = sum(Y_foldEval == yhat)/length(yhat);
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

%%
% cv solution 5667%
% percExp = 0.90;
% numpc =  find((explainedPC > percExp),1);
% 
% % 
% % PC_train = [ones(size(X_train, 1),1) X_train*coeff];
% % PC_eval =  [ones(size(X_eval, 1),1) X_eval*coeff];
% PC_train = [X_train*coeff];
% PC_eval =  [X_eval*coeff];
% PC = X*coeff;
% 
% PC_train = PC_train(:, 1:numpc);
% PC_eval =  PC_eval(:, 1:numpc);
% PC = PC(:, 1:numpc);
% 
% %visualize
% figure(2)
% plot(PC_train(Y_train == 0,2),PC_train(Y_train == 0,3),'r*','markersize',5)
% hold on;
% plot(PC_train(Y_train == 1,2),PC_train(Y_train == 1,3),'bo','markersize',5)
% legend('Male','Female')
% xlabel('PC1')
% ylabel('PC2')


%% Cross-validation on train set to determin minleaf

% minLeaf_CV = floor(linspace(1,size(PC_train,2),10));  %10 different values
% 
% numFolds = 10;
% cvIdx = crossvalind('Kfold', length(Y_train), numFolds);
% 
% for numLeaf = 1:numel(minLeaf_CV)
%     for cv = 1:numFolds
%     
%     
%     end
% end

% add noise with + 0.001.*normrnd(0,1,size(PC_train))
%%%%%%%%%%%%%%%%%%%%%%%%% DEC TREE EVAL
tic;
% mod = TreeBagger(500,PC_train,Y_train,'Method','classification','OOBPred','On','MinLeaf',10,'FBoot',0.5,'NVarToSample',64);
mod = TreeBagger(500,X_train,Y_train,'Method','classification','OOBPred','On','MinLeaf',10,'FBoot',0.5);
toc;
yhat = str2num(cell2mat(predict(mod,X_eval)));

acc = sum(Y_eval == yhat)/length(yhat);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%make predictions
%train with all data
tic;
% mod = TreeBagger(500,PC_train,Y_train,'Method','classification','OOBPred','On','MinLeaf',10,'FBoot',0.5,'NVarToSample',64);
mod = TreeBagger(1000,PC,Y,'Method','classification','OOBPred','On','MinLeaf',10,'FBoot',0.5);
toc;


% PC_test =  [ones(size(X_test, 1),1) X_test*coeff];
PC_test =  [X_test*coeff];

PC_test = PC_test(:, 1:numpc);

yhat = str2num(cell2mat(predict(mod,PC_test)));

% %save model and prediction
save('decTree_mod.mat','mod');
save('yhat_decTree.mat','yhat');

% 
dlmwrite('submit.txt', yhat);





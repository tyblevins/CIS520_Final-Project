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

disp(sprintf('loading in data........\n'))
gender_train = dlmread('genders_train.txt');
% image_raw   = dlmread('images_train.txt');
% image_feats = dlmread('image_features_train.txt');
words_train = dlmread('words_train.txt');

%load dictionary
[rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');

%find stop words
%predefined dictionary
stop_words  = textread('default_stopVoc.txt','%s','delimiter','\n');
rmvStop = find(sum(cell2mat(cellfun(@strcmp,repmat(voc,1,length(stop_words))' , repmat(stop_words,1,size(voc,1)),'UniformOutput',0)),1)' == 1);
words_train(:,rmvStop) = [];

%most frequent
% [wordfreq, freqidx] = sort(sum(words_train,1),'descend');
% stopRemove = 30;
% voc{freqidx(1:stopRemove)}   %show the 10 words which have largest effect on 1st pca
% words_train(:,freqidx(1:stopRemove)) = [];

%done
disp(sprintf('Done Loading Data (stop words above)! \n'))


%assign feats
Y = gender_train;
X = [words_train];

%% train stuff
%define x and y's from data above
numTr = ceil(size(Y,1)*.70);

trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

%define train set
Y_train = Y(trainIdx,:);
X_train = X(trainIdx,:);

%defin evaluation set
Y_eval = Y(evalIdx,:);
X_eval = X(evalIdx,:);


%% transform features
% [coeff,score,latent,tsquared,explained,mu] = pca(X_train,'NumComponents',50);
[coeff,score,latent] = pca(X,'NumComponents',500);


%look at the coef 
[bigvalues, bigidx] = sort(coeff(:,1), 'descend');
voc{bigidx(1:10)}   %show the 10 words which have largest effect on 1st pca

explainedPC = cumsum(latent)./sum(latent);


%% CV to find number of pc to use
PC = [ones(size(X, 1),1) X*coeff];
numParam = 10;
pc_CV = (linspace(0.65,0.70,numParam));  %10 different values
numFolds = 10;
cvIdx = crossvalind('Kfold', length(Y_train), numFolds);

cvAcc = zeros(numFolds,numParam);

disp(sprintf('starting CV........\n'))
for expIdx = 1:numel(pc_CV)
    for cv = 1:numFolds
        numpc =  find((explainedPC > pc_CV(expIdx)),1);
        tmpPC = PC(:, 1:numpc);
        PC_fold     = tmpPC(cvIdx ~= cv,:);
        Y_fold      = Y(cvIdx ~= cv);
        PC_foldEval = tmpPC(cvIdx == cv,:);
        Y_foldEval  = Y(cvIdx == cv);
        tic;
        mod = TreeBagger(250,PC_fold,Y_fold,'Method','classification','OOBPred','On','MinLeaf',1,'FBoot',0.5);
        toc;
        yhat = str2num(cell2mat(predict(mod,PC_foldEval)));
        
        
        cvAcc(cv,expIdx) = sum(Y_foldEval == yhat)/length(yhat);
        disp(['Progress: Parameter - ' num2str(expIdx) '/' num2str(numParam) ...
            ' Fold - ' num2str(cv) '/' num2str(numFolds) sprintf('\n')])

    end
end

parAcc = mean(cvAcc,1);
[~, bestIdx] = max(parAcc);

figure(1)
plot(pc_CV,parAcc);

percExp = pc_CV(bestIdx);
numpc =  find((explainedPC > percExp),1);

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

% numpc =  find((explainedPC > .80),1);


PC_train = X_train * coeff;
PC_eval =  X_eval * coeff;

PC_train = PC_train(:, 1:numpc);
PC_eval =  PC_eval(:, 1:numpc);

%visualize
figure(2)
plot(PC_train(Y_train == 0,2),PC_train(Y_train == 0,3),'r*','markersize',5)
hold on;
plot(PC_train(Y_train == 1,2),PC_train(Y_train == 1,3),'bo','markersize',5)
legend('Male','Female')
xlabel('PC1')
ylabel('PC2')


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

%%%%%%%%%%%%%%%%%%%%%%%%% DEC TREE
tic;
mod = TreeBagger(250,PC_train,Y_train,'Method','classification','OOBPred','On','MinLeaf',1,'FBoot',0.5);
toc;
yhat = str2num(cell2mat(predict(mod,PC_eval)));

acc = sum(Y_eval == yhat)/length(yhat);

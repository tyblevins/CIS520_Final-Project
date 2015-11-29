%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% AdaBoost
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
% words_train(:,featSel_resultsKNN == 0) = [];
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


%% TEST DATA
% image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');

% words_test(:,featSel_resultsKNN == 0) = [];
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

%%%%%%%%%%%%%%%%%%%%%%%%% ADABOOST EVAL
T = 500;
disp('Training...')
tic;
% mod = TreeBagger(500,PC_train,Y_train,'Method','classification','OOBPred','On','MinLeaf',10,'FBoot',0.5,'NVarToSample',64);
adaStump = fitensemble(X,Y,'AdaBoostM1',T,'Tree');
toc;
figure;
plot(resubLoss(adaStump,'Mode','Cumulative'));
xlabel('Number of stumps');
ylabel('Training error');
legend('AdaBoost');

disp('Cross-Validation')
cvada = crossval(adaStump,'KFold',5);
figure;
plot(kfoldLoss(cvada,'Mode','Cumulative'));
xlabel('Ensemble size');
ylabel('Cross-validated error');
legend('AdaBoost');

cada = compact(adaStump);
cada = removeLearners(cada,450:cada.NTrained);
accLoss_check = loss(cada,X_train,Y_train)

%calc Train errory

tic;
yhat_train = (predict(cada,X));
toc;
% 
accTrain = sum(Y == yhat_train)/length(yhat_train)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%make predictions
%train with all data

tic;
yhat = (predict(cada,X_test));
toc;

% %save model and prediction
save('ada_mod.mat','cada');
save('yhat_ada.mat','yhat');
save('yhat_adaTrain.mat','yhat_train');

% 
% dlmwrite('submit.txt', yhat);

%Jared Wilson
%11/3/2015
% CIS520 -- Final Project Fall 2015
% featsel ON FULL Vocabulary
% NOTE:
% This script should be run from folder svm_v1

clear;

%% Load in data
addpath(genpath('..\CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('featSel'))

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
% 
% words_test(:,rmvStop) = [];
% words_test(:,notUsed_Idx) = [];

disp(sprintf('Done Loading Data \n'))


%assign feats
Y = gender_train;
X = [words_train];
X_test = [words_test];

Y(Y == 0) = -1;


%mean center data
X = X + 2e-13;
avgX = mean(X,1);
stdX = std(X)+2e-13;

X = bsxfun(@rdivide ,((X)  - repmat(avgX,size(X,1),1)), stdX);
X_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);

% X = X + 0.05.*normrnd(0,1,size(X));  %add noise

%% train stuff
%define x and y's from data above
% numTr = ceil(size(Y,1)*.70);
numTr = 1000;   %only concider first 1000 to avoid overfitting


trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

%define train set
Y(Y == 0) = -1;  %change labels to -1/1 for peceptron learning

Y_train = Y(trainIdx,:);
X_train = X(trainIdx,:);

%defin evaluation set
Y_eval = Y(evalIdx,:);
X_eval = X(evalIdx,:);


%L1 NORM
% lambda = linspace(1,10e3,20);
% 
% for i = 1:numel(lambda)
%     testLam = lambda(i);
%     lassoEst = @(w) (sum((Y - X*w).^2)) + lambda * (sum(abs(w)));
%     what_lasso = fminsearch(lassoEst, ones(1,size(X,2)));
%     disp(num2str(i));
% end




%% stepwise feature selection

%initilize knn search
k = 13;
X_step_knn      = X_train(:,1);  %start with the first feature
mseStep_knn     = zeros(size(X,2),1);
numRmv_knn      = 0;
featSel_resultsKNN = ones(size(X_train,2),1);

%K-nn loss function
mdl = fitcknn(X_step_knn,Y_train,'NumNeighbors', k);
yhat_knn = predict(mdl,X_step_knn);
mseStep_knn(1) = sum(Y_train ~= yhat_knn)/length(yhat_knn);

%initialize L0 search
X_step_lz         = X_train(:,1);  %start with the first feature
mseStep_lz        = zeros(size(X,2),1);
what_lz           = cell(size(X_train,2),1);
numRmv_lz         = 0;
featSel_resultsLZ = ones(size(X_train,2),1);
%calculate the first step prior to loop
%first step regression
lzEst = @(w) sum((Y_train ~= sign(X_step_lz*w)));
what_lz{1} = fminsearch(lzEst, ones(1,size(X_step_lz,2))');
mseStep_lz(1) = lzEst(what_lz{1});  

%L2-error regression search loss function
% what_lz{1} = mldivide((X_step'*X_step),(X_step'*Y_train));


tic;
for i = 2:size(X_train,2);
    X_step_knn      = [X_step_knn X_train(:,i)];  %start with the first feature
    X_step_lz      = [X_step_lz X_train(:,i)];  %start with the first feature

    %K-nn loss function
    mdl = fitcknn(X_step_knn,Y_train,'NumNeighbors', k);
    yhat_knn = predict(mdl,X_step_knn);
    mseStep_knn(i) = sum(Y_train ~= yhat_knn)/length(yhat_knn);

    if(mseStep_knn(i) >= mseStep_knn(i-1))
        X_step_knn(:,end) = [];  %remove the last feature added
        numRmv_knn = numRmv_knn + 1;
        featSel_resultsKNN(i) = 0; %feature removed
    end
    
    
    % L0 regression loss
    lzEst = @(w) sum((Y_train ~= sign(X_step_lz*w)));
    what_lz{i} = fminsearch(lzEst, ones(1,size(X_step_lz,2))');
    mseStep_lz(i) = lzEst(what_lz{i});  
    
   if(mseStep_lz(i) >= mseStep_lz(i-1))
        X_step_lz(:,end) = [];  %remove the last feature added
        numRmv_lz = numRmv_lz + 1;
        featSel_resultsLZ(i) = 0; %feature removed
    end

    disp(['Step: ' num2str(i) '/' num2str(size(X_train,2)) sprintf('\n') ...
        ' featsElim KNN: ' num2str(numRmv_knn) ...
        ' featsElim L0: ' num2str(numRmv_lz) sprintf('\n')])

end
totTime = toc;


wordsKNN = voc{featSel_resultsKNN == 1};
wordsLZ = voc{featSel_resultsLZ == 1};   

save('L0_vocSel.mat','featSel_resultsLZ');
save('knn_vocSel.mat','featSel_resultsKNN');


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


%Jared Wilson 
%ensemble attempt
clear;
addpath(genpath('ensemble'))


Y = dlmread('genders_train.txt');
Y(Y == 0) = -1;
%% TRAIN ENSEMBLE MODEL

%creat a train model
yhatTr_dec  = load('yhat_decTreeTrain.mat');
yhatTr_dec = yhatTr_dec.yhat_train;
yhatTr_decLasso = load('yhat_decLassoTr.mat');
yhatTr_decLasso = yhatTr_decLasso.yhat_train;

yhatTr_perc = load('yhat_percTr.mat');
yhatTr_perc = yhatTr_perc.yhat_train;

yhatTr_Lin = load('yhat_liblinearTrain.mat');
yhatTr_Lin = yhatTr_Lin.yhat_train;
yhatTr_LinLasso = load('yhat_liblinLassoTrain.mat');
yhatTr_LinLasso = yhatTr_LinLasso.yhat_train;

yhatTr_svm = load('yhat_svmTr.mat');
yhatTr_svm = yhatTr_svm.yhat_train;
yhatTr_svmLasso = load('yhat_svmLassoTr.mat');
yhatTr_svmLasso = yhatTr_svmLasso.yhat_train;

yhatTr_ada = load('yhat_adaTrain.mat');
yhatTr_ada = yhatTr_ada.yhat_train;
yhatTr_adaLasso = load('yhat_adaLassoTrain.mat');
yhatTr_adaLasso = yhatTr_adaLasso.yhat_train;

% X = [yhatTr_dec yhatTr_perc yhatTr_Lin yhatTr_svm yhatTr_ada];
% X = [yhatTr_dec  yhatTr_Lin  yhatTr_ada yhatTr_svmLasso];

X = [yhatTr_decLasso  yhatTr_LinLasso  yhatTr_adaLasso yhatTr_svmLasso ];
% X = [yhatTr_decLasso  yhatTr_Lin  yhatTr_adaLasso yhatTr_svm ];


X(X == 0) = -1;

%split
numTr = ceil(size(Y,1)*.70);


trainIdx = randperm(size(Y,1),numTr);
evalIdx = 1:size(Y,1);
evalIdx(trainIdx) = [];

%define train set
Y_train = Y(trainIdx,:);
X_train = X(trainIdx,:);

%defin evaluation set
Y_eval = Y(evalIdx,:);
X_eval = X(evalIdx,:);


lzEst = @(w) sum((Y_train ~= sign(X_train*w)));
wEns = fminsearch(lzEst, ones(1,size(X_train,2))');

% randInit = rand(size(X_train,2),1);
% wEns = fminsearch(lzEst, randInit);


% lzEst = @(w) sum((Y_train - (X_train*w)).^2);
% wEns_eval = fminsearch(lzEst, 0.5*ones(1,size(X_train,2))');


yhat_comb = sign(X_eval*wEns);

acc = mean(Y_eval == yhat_comb)

% lzEst = @(w) sum((Y - (X*w)).^2);
% wEns = fminsearch(lzEst, 0.1*ones(1,size(X_train,2))');

lzEst = @(w) sum((Y ~= sign(X*w)));
wEns = fminsearch(lzEst, 0.25*ones(1,size(X,2))');

% wEns = [1.05; 1; 1; 1]


yhat_comb = sign(X*wEns);

acc = mean(Y == yhat_comb)

%custom weights

%% TEST MODEL
yhat_dec  = load('yhat_decTree.mat');
yhat_dec = yhat_dec.yhat;
yhat_decLasso = load('yhat_decLasso.mat');
yhat_decLasso = yhat_decLasso.yhat;

yhat_perc = load('yhat_perc.mat');
yhat_perc = yhat_perc.yhat;

yhat_Lin = load('yhat_liblinear.mat');
yhat_Lin = yhat_Lin.yhat;
yhat_LinLasso = load('yhat_liblinLasso.mat');
yhat_LinLasso = yhat_LinLasso.yhat;

yhat_svm = load('yhat_svm.mat');
yhat_svm = yhat_svm.yhat;
yhat_svmLasso = load('yhat_svmLasso.mat');
yhat_svmLasso = yhat_svmLasso.yhat;

yhat_ada = load('yhat_ada.mat');
yhat_ada = yhat_ada.yhat;
yhat_adaLasso = load('yhat_adaLasso.mat');
yhat_adaLasso = yhat_adaLasso.yhat;

% X_test = [yhat_dec yhat_perc yhat_Lin yhat_svm yhat_ada];

X_test = [yhat_decLasso  yhat_LinLasso  yhat_adaLasso yhat_svmLasso];

X_test(X_test == 0) = -1;


yhat_submit = sign(X_test*wEns);

sum(yhat_submit == 0)


%for those that had even vote just guess
% RandomGuesses=round(rand(sum(yhat_submit == 0),1));
% yhat_submit(yhat_submit == 0)=RandomGuesses;

yhat_submit(yhat_submit == -1 ) = 0;

dlmwrite('submit.txt', yhat_submit);


%create confusion matrix to show percent aggree between each model
yhat_NEG = X;

agreeMAT = yhat_NEG'*yhat_NEG;
agreeMAT = abs(agreeMAT - size(X,1))./2;

figure(1)
imagesc(agreeMAT);
colorbar;
set(gca,'XTick',[1:4])
set(gca,'YTick',[1:4])
set(gca,'XTickLabel',{'DecTree', 'Log Reg', 'Ada','SVM'})
set(gca,'YTickLabel',{'DecTree', 'Log Reg', 'Ada','SVM'})

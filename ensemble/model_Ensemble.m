%Jared Wilson 
%ensemble attempt
clear;

Y = dlmread('genders_train.txt');
Y(Y == 0) = -1;
%% TRAIN ENSEMBLE MODEL

%creat a train model
yhatTr_dec  = load('yhat_decTreeTrain.mat');
yhatTr_dec = yhatTr_dec.yhat_train;
yhatTr_perc = load('yhat_percTr.mat');
yhatTr_perc = yhatTr_perc.yhat_train;
yhatTr_Lin = load('yhat_liblinearTrain.mat');
yhatTr_Lin = yhatTr_Lin.yhat_train;
yhatTr_svm = load('yhat_svmTr.mat');
yhatTr_svm = yhatTr_svm.yhat_train;

X = [yhatTr_dec yhatTr_perc yhatTr_Lin yhatTr_svm];
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
wEns = fminsearch(lzEst, 0.5*ones(1,size(X_train,2))');

yhat_comb = sign(X_eval*wEns);

acc = mean(Y_eval == yhat_comb)


%% TEST MODEL
yhat_dec  = load('yhat_decTree.mat');
yhat_dec = yhat_dec.yhat;
yhat_perc = load('yhat_perc.mat');
yhat_perc = yhat_perc.yhat;
yhat_Lin = load('yhat_liblinear.mat');
yhat_Lin = yhat_Lin.yhat;
yhat_svm = load('yhat_svm.mat');
yhat_svm = yhat_svm.yhat;


X_test = [yhat_dec yhat_perc yhat_Lin yhat_svm];
X_test(X_test == 0) = -1;


yhat_submit = sign(X_test*wEns);

sum(yhat_submit == 0)


%for those that had even vote just guess
% RandomGuesses=round(rand(sum(yhat_submit == 0),1));
% yhat_submit(yhat_submit == 0)=RandomGuesses;

yhat_submit(yhat_submit == -1 ) = 0;

dlmwrite('submit.txt', yhat_submit);


%create confusion matrix to show percent aggree between each model
yhat_NEG = yhat_ens;
yhat_NEG(yhat_ens == 0) = -1;

agreeMAT = yhat_NEG'*yhat_NEG;
agreeMAT = abs(agreeMAT - length(yhat_knn))./2;

figure(1)
imagesc(agreeMAT);
colorbar;
set(gca,'XTick',[1:3])
set(gca,'YTick',[1:3])
set(gca,'XTickLabel',{'K-NN','DecTree', 'Perceptron'})
set(gca,'YTickLabel',{'K-NN','DecTree', 'Perceptron'})

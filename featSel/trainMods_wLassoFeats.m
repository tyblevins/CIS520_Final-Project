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
addpath(genpath('logistic_reg'))

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
Xnorm = X + 2e-13;
avgX = mean(X,1);
stdX = std(X)+2e-13;
Xnorm = bsxfun(@rdivide ,((Xnorm)  - repmat(avgX,size(Xnorm,1),1)), stdX);
Xnorm_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);

Xstats = [avgX; stdX];
save('Xstats.mat','Xstats');
%elatstic net predictions
wlasso = lassoResults{1};
wlasso(lassoRmv) = []; 
lasso_trainPred = lassoResults{2} +  Xnorm*wlasso;

yhatTr_lasso = sign(lasso_trainPred);
acc_train = mean(Y == yhatTr_lasso)



%%%%%%%%%%%%%%%%%%%%%%%%% SVM CROSS VALIDATION
%# grid of parameters
folds = 5;
% [C,gamma] = meshgrid(6:1:12,-22:2:-10);

[C,gamma] = meshgrid(-2:1:6,-16:2:0);

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
parfor i=1:numel(C)
    cv_acc(i) = svmtrain(Y, Xnorm, ...
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

% unnormalized settings
% tic
% cv_accTest =  svmtrain(Y, X, sprintf('-t 2 -c 256 -g 0.00000095367 -h 1 -v 5'));
% toc;

tic
cv_accTest =  svmtrain(Y, Xnorm, sprintf('-t 2 -c 2 -g 0.00097656 -h 1 -v 5'));
toc;


tic
mod =  svmtrain(Y, Xnorm, sprintf('-t 2 -c 2 -g 0.00097656 -h 1'));
toc;

[yhat acc vals] = svmpredict(ones(size(Xnorm_test,1),1), Xnorm_test, mod);
[yhat_train train_acc vals] = svmpredict(Y, Xnorm, mod);


%save model and prediction
save('svmLasso_mod.mat','mod');
save('yhat_svmLasso.mat','yhat');
save('yhat_svmLassoTr.mat','yhat_train');

%%
% Logistic Regression Train
%# grid of parameters
folds = 5;
% [C,epsilon] = meshgrid(2:2:8, -10:2:-6);  not normalized grid search
[C,epsilon] = meshgrid(-8:1:2, -12:2:4);

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
parfor i=1:numel(C)
    cv_acc(i) = train(Y, sparse(Xnorm), ...
                    [sprintf('-s 0 -c %f -e %f -B 1 -v %d', 2^C(i), 2^epsilon(i), folds), 'col']);
end

%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);

%# contour plot of paramter selection
contour(C, epsilon, reshape(cv_acc,size(C))), colorbar
hold on
plot(C(idx), epsilon(idx), 'rx')
text(C(idx), epsilon(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
    'HorizontalAlign','left', 'VerticalAlign','top')
hold off
xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')

acc_test = train(Y, sparse(Xnorm), ['-s 0 -c 0.0078 -e 0.0625 -B 1 -v 5', 'col']);


%% Train full model
% not normalized
% model = train(Y, sparse(X), ['-s 0 -c 16 -e 0.0039 -B 1', 'col']);
% [yhat] = predict(round(rand(size(X_test,1),1)), sparse(X_test), model, ['-q', 'col']);
% [yhat_train] = predict(round(rand(size(X,1),1)), sparse(X), model, ['-q', 'col']);

model = train(Y, sparse(Xnorm), ['-s 0 -c 0.0078 -e 0.0625 -B 1', 'col']);
[yhat] = predict(round(rand(size(Xnorm_test,1),1)), sparse(Xnorm_test), model, ['-q', 'col']);
[yhat_train] = predict(round(rand(size(Xnorm,1),1)), sparse(Xnorm), model, ['-q', 'col']);

trainAcc = sum(Y == yhat_train)/length(yhat_train)

save('liblinLasso_mod.mat','model');
save('yhat_liblinLasso.mat','yhat');
save('yhat_liblinLassoTrain.mat','yhat_train');


%%
% Random Forest Train
Y(Y == -1) = 0;
tic;
% mod = TreeBagger(500,PC_train,Y_train,'Method','classification','OOBPred','On','MinLeaf',10,'FBoot',0.5,'NVarToSample',64);
% rndForest = TreeBagger(500,X,Y,'Method','classification','OOBPred','On','MinLeaf',2,'FBoot',0.25,'NVarToSample',25);
rndForest = TreeBagger(500,X,Y,'Method','classification','OOBPred','On','MinLeaf',2,'FBoot',0.75);
toc;

figure;
oobErrorBaggedEnsemble = oobError(rndForest);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

yhat_train = str2num(cell2mat(predict(rndForest,X)));
acc_train = mean(Y == yhat_train)

yhat = str2num(cell2mat(predict(rndForest,X_test)));

%save model and prediction
save('decLasso_mod.mat','rndForest');
save('yhat_decLasso.mat','yhat');
save('yhat_decLassoTr.mat','yhat_train');


toc;

%% 
% Ada Boost Train
T = 1000;
disp('Training...')
tic;

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
cada = removeLearners(cada,500:cada.NTrained);
accLoss_check = loss(cada,X,Y)

%calc Train error
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
save('adaLasso_mod.mat','cada');
save('yhat_adaLasso.mat','yhat');
save('yhat_adaLassoTrain.mat','yhat_train');
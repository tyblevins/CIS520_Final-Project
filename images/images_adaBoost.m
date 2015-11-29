%Read in training data to workspace
%Jared Wilson
%11/17/2015
clear;
close all;


addpath(genpath('../CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('images'))
addpath(genpath('tools'))

gender_train = dlmread('genders_train.txt');
image_raw_train   = dlmread('images_train.txt');
image_raw_test = dlmread('images_test.txt');
% words_train = dlmread('words_train.txt');

%load dictionary
% [rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');

%use test and train
img_X = [image_raw_train; image_raw_test];
imgAvg = mean(img_X,1);

grayRaw = zeros(size(img_X,1),100,100);

for i=1:size(img_X,1)
  cur_row=img_X(i,:);
  cur_img=reshape(cur_row,[100 100 3]);
%     
        grayRaw(i,:,:) = rgb2gray(uint8(cur_img));

%     imshow(uint8(squeeze(grayRaw(i,:,:))));
%     disp(num2str(i))
%   pause(3);
end

grayFlat = reshape(grayRaw, size(grayRaw,1),size(grayRaw,2)*size(grayRaw,3));  %flatten

numTr = size(image_raw_train,1);


Y = gender_train;
X = [grayFlat(1:numTr,:)];
X_test = [grayFlat(numTr+1:end,:)];


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


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

cellSize = 15;
numCells = floor([100 100]/cellSize);
lbpFeatures = zeros(size(img_X,1),numCells(1)*numCells(2));

Y = gender_train;
Y(Y == 0) = -1;  %change labels to -1/1 for peceptron learning

for i=1:size(img_X,1)
  cur_row=img_X(i,:);
  cur_img=reshape(cur_row,[100 100 3]);
%     
        grayRaw(i,:,:) = rgb2gray(uint8(cur_img));

%     imshow(uint8(squeeze(grayRaw(i,:,:))));
%     disp(num2str(i))
%   pause(3);
end

%compute mean image
grayAvg = zeros(100,100);
for i = 1:size(grayAvg,1)
    for j = 1:size(grayAvg,2)
        grayAvg(i,j) = mean(grayRaw(:,i,j));
    end
end
imshow(uint8(grayAvg));

numTr = size(image_raw_train,1);

gray_train = grayRaw(1:numTr,:,:);
gray_test = grayRaw(numTr+1:end,:,:);

%remove mean from each picture then flatten
normGray = grayRaw - repmat(reshape(grayAvg,[1 100 100]), [size(grayRaw,1) 1 1]);
normGrayFlat = reshape(normGray, size(normGray,1),size(normGray,2)*size(normGray,3));  %flatten

grayFlat = reshape(grayRaw, size(grayRaw,1),size(grayRaw,2)*size(grayRaw,3));  %flatten

maleGray   = mean(grayFlat(Y == -1,:),1);
maleGray   = reshape(maleGray,[100 100 1]);
figure(44)
subplot(121)
imshow(uint8(maleGray));

femaleGray = mean(grayFlat(Y == 1,:),1);
femaleGray   = reshape(femaleGray,[100 100 1]);
subplot(122)
imshow(uint8(femaleGray));


numTr = size(image_raw_train,1);

normG_train = normGray(1:numTr,:,:);
normG_test = normGray(numTr+1:end,:,:);

gray_train = grayFlat(1:numTr,:);
gray_test = grayFlat(numTr+1:end,:);

numNeig = 8;
cellSize = [10 10];
numCells = prod(floor(size(squeeze(gray_train(1,:,:)))./cellSize));

% N = numCells * (numNeig + 2);  %set for upright false
N = numCells * ((numNeig * (numNeig - 1)) + 3);  %set for upright true

lbp_train = zeros(size(gray_train,1),N);
tic;
%calculate lpbfeats for all images
for i = 1:size(gray_train,1)
%     tic;
    lbp_train(i,:) = extractLBPFeatures(squeeze(normG_train(i,:,:)),'CellSize',cellSize, ...
        'NumNeighbors' ,numNeig,'upright',true);
%     toc;
end
toc;

lbp_male = mean(lbp_train(Y == -1,:),1);
lbp_female = mean(lbp_train(Y == 1,:),1);

lbp_genDif = sum((lbp_male - lbp_female));


%define training stuff
numTr = ceil(size(Y,1)*.70);

trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

X = normGray;

Y_train = Y(trainIdx,:);
X_train = X(trainIdx,:);

%defin evaluation set
Y_eval = Y(evalIdx,:);
X_eval = X(evalIdx,:);

model = svmtrain(Y_train, X_train, sprintf('-t 3 -c 10'));
[yhat acc vals] = svmpredict(Y_eval, X_eval, model);
test_acc = mean(yhat==Y_eval);

%%%%%%%%%%%%%%%%%
%% FDA Algorithm
tic;
[Z_norm,W_norm] = FDA(normG_train',Y,100);
toc;

fda_train = normG_train*W;
fda_test  = normG_test*W;

model = svmtrain(Y, fda_train, sprintf('-t 0 -c 0.1'));

[W,H] = nnmf(normGrayFlat,50);

numTr = size(image_raw_train,1);

W_train = W(1:numTr,:);

avgMale_W   = mean(W_train(gender_train == 0, :) ,1);
avgFemale_W =  mean(W_train(gender_train == 1, :) ,1);

%L2 distance between classes
distL2 = sum((avgMale_W - avgFemale_W).^2);

numTr = ceil(size(Y,1)*.70);

trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

figure(40)
numRec = 35;
%compare avg male and avg female
male_rec = (avgMale_W(1:numRec)*H(1:numRec,:));
    cur_img=reshape(male_rec,[100 100 1]);
    subplot(121)
    imshow(uint8(cur_img));
female_rec = (avgFemale_W(1:numRec)*H(1:numRec,:));
    cur_img=reshape(male_rec,[100 100 1]);
    subplot(122)
    imshow(uint8(cur_img));

    
%%%%%%%%%%%%%%%%%%%%%%%% TRAIN NNMF MODEL
tic;

    k = 13;
    mdl = fitcknn(W_train,Y,'NumNeighbors', k);
    yhat_knn = predict(mdl,W_train);

    acc_Knn = sum(Y == yhat_knn)/length(yhat_knn)


%      model = svmtrain(Y, W_train, sprintf('-t 0 -c 0.1'));
%     [yhatnnmf accnnmf valsnnmf] = svmpredict(Y, W_train, model);
    
timeSVM_nnmf = toc;
    

%%%%%%%%%%%%%%%%%%%%%%%%%%% Eigenface stuff

tic;
[coeff,score,latent,trash,trash,mu] = pca(normGrayFlat ,'NumComponents',200);
toc;
% grayCov = normGrayFlat'*normGrayFlat;

explainedPC = cumsum(latent)./sum(latent);
percExp = 0.90;
numpc =  find((explainedPC > percExp),1);

PC = score(:,1:numpc);

projC = coeff(:,1:numpc);

numTr = size(image_raw_train,1);

PC_train = PC(1:numTr,:);

numTr = ceil(size(Y,1)*.70);

trainIdx = 1:numTr;
evalIdx  = (numTr+1):size(gender_train,1);

PC_eval  = PC_train(evalIdx,:);
PC_train = PC_train(trainIdx,:);

avgMale   = mean(PC_train(gender_train == 0, :) ,1);
avgFemale =  mean(PC_train(gender_train == 1, :) ,1);

%visualize
figure(22)
plot(PC_train(Y_train == -1,1),PC_train(Y_train == -1,2),'r*','markersize',5)
hold on;
plot(PC_train(Y_train == 1,1),PC_train(Y_train == 1,2),'bo','markersize',5)
legend('Male','Female')
xlabel('PC1')
ylabel('PC2')




%compare avg male and avg female
male_recPC = (avgMale*projC');
    cur_img=reshape(male_rec,[100 100 1]);
    imshow(uint8(cur_img));
    pause(1);

%  model = svmtrain(Y, PC_train, sprintf('-t 0 -c 0.001 -h 1'));
 model = svmtrain(Y, PC_train, sprintf('-t 0 -c 0.1'));

[yhatPC accPC valsPC] = svmpredict(Y, PC_train, model);
   
    
    
%EXTRA STFF
% show nowrmalized image
% for i=1:size(normGray,1)
%     imshow(uint8(squeeze(normGray(i,:,:))));
%       pause(3);
% 
% end

% 
% corr_offset = zeros(size(grayRaw,1),2);
% for i=1:size(grayRaw,1)
%     cc = xcorr2(imgAvg , squeeze(grayRaw(i,:,:)));
%     [max_cc, imax] = max(abs(cc(:)));
%     [ypeak, xpeak] = ind2sub(size(cc),imax(1));
%     corr_offset(i,:) = [(ypeak-size(squeeze(grayRaw(i,:,:)),1)) (xpeak-size(squeeze(grayRaw(i,:,:)),2))];
% 
% end
% 
% 
% C = conv2(imgAvg , squeeze(grayRaw(2,:,:)),'same');
% [maxVal, loc] = max(C(:));
% [row,col] = ind2sub(size(C),loc);
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calculate non-negative matrix factorization
% [W,H] = nnmf(image_raw,100);

% %reconstruct images look at the frist one
% imgIdx = 1;
% for i = [20]
%     image_rec = (W(imgIdx,1:i)*H(1:i,:));
%     cur_row = image_rec(imgIdx,:);
%     cur_img=reshape(cur_row,[100 100 1]);
%     imshow(uint8(cur_img));
%     pause(1);
% end
% 
% save('nnmfH_100.mat','H');
% save('nnmfW_100.mat','W');
% %% visualize the eigen vectors 
% 
% % img_X = H(1,:);
% img_X = H;
% %let look at H matrix
% for i=1:size(img_X,1)
%   cur_row=img_X(i,:).*(255/max(img_X(i,:)));
%   cur_img=reshape(cur_row,[100 100 3]);
%   imshow(uint8(cur_img));
%   pause(3);
% end

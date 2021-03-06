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

Y = gender_train;
Y(Y == 0) = -1;  %change labels to -1/1 for peceptron learning

for i=1:size(img_X,1)
  cur_row=img_X(i,:);
  cur_img=reshape(cur_row,[100 100 3]);
  

% FACE FIND STUFF -- takes too long
%     [face_a,skin_region]=face(cur_img);
%     imshow(uint8(face_a));

%     if(sum(sum(any(face_a))))
%         grayRaw(i,:,:) = rgb2gray(uint8(face_a));
%     else
%         grayRaw(i,:,:) = rgb2gray(uint8(cur_img));
%     end
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



%remove mean from each picture then flatten
normGray = grayRaw - repmat(reshape(grayAvg,[1 100 100]), [size(grayRaw,1) 1 1]);
normGrayFlat = reshape(normGray, size(normGray,1),size(normGray,2)*size(normGray,3));  %flatten

numTr = size(image_raw_train,1);

normG_train = normGrayFlat(1:numTr,:);
normG_test = normGrayFlat(numTr+1:end,:);


tic;
[Z,W] = FDA(normG_train',Y);
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

avgMale   = mean(PC_train(gender_train == 0, :) ,1);
avgFemale =  mean(PC_train(gender_train == 1, :) ,1);

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

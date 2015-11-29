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

tic
for i=1:size(img_X,1)
  cur_row=img_X(i,:);
  cur_img=reshape(cur_row,[100 100 3]);
%     
        grayRaw(i,:,:) = rgb2gray(uint8(cur_img));

%     imshow(uint8(squeeze(grayRaw(i,:,:))));
%     disp(num2str(i))
%   pause(3);
end
toc

%compute mean image
grayAvg = zeros(100,100);
for i = 1:size(grayAvg,1)
    for j = 1:size(grayAvg,2)
        grayAvg(i,j) = mean(grayRaw(:,i,j));
    end
end
imshow(uint8(grayAvg));

numTr = size(image_raw_train,1);



%remove mean from each picture then flatten
normGray = grayRaw - repmat(reshape(grayAvg,[1 100 100]), [size(grayRaw,1) 1 1]);
normGrayFlat = reshape(normGray, size(normGray,1),size(normGray,2)*size(normGray,3));  %flatten

grayFlat = reshape(grayRaw, size(grayRaw,1),size(grayRaw,2)*size(grayRaw,3));  %flatten


[Wgray,Hgray] = nnmf(normGrayFlat,100);

%reconstruct images look at the frist one
imgIdx = 1;
for i = 80
    image_rec = (Wgray(imgIdx,50:i)*Hgray(50:i,:));
    cur_row = image_rec(imgIdx,:);
    cur_img=reshape(cur_row,[100 100 1]);
    imshow(uint8(cur_img));
    pause(1);
end




numTr = size(image_raw_train,1);


grayflat_train = grayRaw(1:numTr,:,:);
gray_test = grayRaw(numTr+1:end,:,:);

normG_train = normGray(1:numTr,:,:);
normG_test = normGray(numTr+1:end,:,:);




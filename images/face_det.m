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
img_X = [image_raw_train];
imgAvg = mean(img_X,1);


grayRaw = zeros(size(img_X,1),100,100);

cellSize = 15;
numCells = floor([100 100]/cellSize);
lbpFeatures = zeros(size(img_X,1),numCells(1)*numCells(2));

Y = gender_train;
Y(Y == 0) = -1;  %change labels to -1/1 for peceptron learning

load face_p146_small.mat
% 5 levels for each octave
model.interval = 5;
% set up the threshold
model.thresh = min(-0.65, model.thresh);

% detector = vision.CascadeObjectDetector('FrontalFaceLBP','ScaleFactor',4);

testDet = cell(size(Y));
for i=1:size(img_X,1)
  cur_row=img_X(i,:);
  cur_img=reshape(cur_row,[100 100 3]);
  


        grayRaw(i,:,:) = rgb2gray(uint8(cur_img));

%     imshow(uint8(squeeze(grayRaw(i,:,:))));
%     disp(num2str(i))
%   pause(3);
end

tic;
bs = detect(squeeze(grayRaw(i,:,:)), model, model.thresh);
bs = clipboxes(squeeze(grayRaw(i,:,:)), bs);
bs = nms_face(bs,0.3);
dettime = toc;


% tic;
% testDet = step(detector,cur_img);
% toc;

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

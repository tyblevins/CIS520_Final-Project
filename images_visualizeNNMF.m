%Read in training data to workspace
%Jared Wilson
%11/17/2015
clear;
close all;


addpath(genpath('../CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))
addpath(genpath('images'))

gender_train = dlmread('genders_train.txt');
img_X   = dlmread('images_train.txt');
img_XTest = dlmread('image_features_train.txt');
% words_train = dlmread('words_train.txt');

%load dictionary
% [rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');

grayRaw = zeros(size(gender_train,1),100,100);
for i=1:size(img_X,1)
  cur_row=img_X(i,:);
  cur_img=reshape(cur_row,[100 100 3]);
  


%   [face_a,skin_region]=face(cur_img);
%   imshow(uint8(face_a));

    grayRaw(i,:,:) = rgb2gray(uint8(cur_img));
      % run the detector
    pos = fdlibmex(squeeze(grayRaw(i,:,:)));

%   imshow(uint8(squeeze(grayRaw(i,:,:))));
  pause(3);
end
[face_a,skin_region]=face(I);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calculate non-negative matrix factorization
[W,H] = nnmf(image_raw,100);

%reconstruct images look at the frist one
imgIdx = 1;
for i = [20]
    image_rec = (W(imgIdx,1:i)*H(1:i,:));
    cur_row = image_rec(imgIdx,:);
    cur_img=reshape(cur_row,[100 100 3]);
    imshow(uint8(cur_img));
    pause(1);
end

save('nnmfH_100.mat','H');
save('nnmfW_100.mat','W');
%% visualize the eigen vectors 

% img_X = H(1,:);
img_X = H;
%let look at H matrix
for i=1:size(img_X,1)
  cur_row=img_X(i,:).*(255/max(img_X(i,:)));
  cur_img=reshape(cur_row,[100 100 3]);
  imshow(uint8(cur_img));
  pause(3);
end

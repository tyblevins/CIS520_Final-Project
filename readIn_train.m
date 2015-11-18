%Read in training data to workspace
%Jared Wilson
%11/17/2015

addpath(genpath('CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))

gender_train = dlmread('genders_train.txt');
image_raw   = dlmread('images_train.txt');
image_feats = dlmread('image_features_train.txt');
words_train = dlmread('words_train.txt');

%load dictionary
[rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% visualize data

% check if words are sparse NOT SPARSE 36% have values
% sparse_check = (words_train ~= 0);
% percent_nonzeros = sum(sum(sparse_check,1),2)/(4998*5000);


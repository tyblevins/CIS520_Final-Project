%Read in testing data to workspace


addpath(genpath('CIS520_twitter_data'))
addpath(genpath('CIS520_Final-Project'))


image_raw_test   = dlmread('images_test.txt');
image_feats_test = dlmread('image_features_test.txt');
words_test = dlmread('words_test.txt');

%load dictionary
[rank, voc] = textread('voc-top-5000.txt','%u %s','delimiter','\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% visualize data

% check if words are sparse NOT SPARSE 36% have values
% sparse_check = (words_train ~= 0);
% percent_nonzeros = sum(sum(sparse_check,1),2)/(4998*5000);


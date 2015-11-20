%% a generative method

[score_train,score_test, ~] = pca_getpc(words_train,words_test);
numpc=50;
PredictedLabel=GaussianNaiveBayes(score_train(:,1:numpc),gender_train,score_test(:,1:numpc));

%% a discriminative method
addpath('./liblinear');
[PredictedLabel] = logistic(words_train, gender_train, words_test);

%% An instance based method
PredictedLabel=KNearestNeighbor(words_train,gender_train,words_test,4);
%% regularization method
[NoisyWords]=AddNoise(words_train,.1);
[PredictedLabel] = logistic(NoisyWords, gender_train , words_test);
%% semi supervised dimensionality reduction of the data

[ score_train,score_test, numpc] = pca_getpc(words_train,words_test);
[PredictedLabel] = logistic( score_train(:, 1:numpc), gender_test, score_test(:,1:numpc));
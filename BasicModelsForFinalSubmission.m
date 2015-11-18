%% Setup
cd('../train')
[Gender, ImageFeatures, Images, Words]=LoadTrainingData;
cd('../test')
[ImageFeaturesTest, ImagesTest, WordsTest]=LoadTestingData;
cd('../code')



%% a generative method

[score_train,score_test, ~] = pca_getpc(Words,WordsTest);
numpc=50;
PredictedLabel=GaussianNaiveBayes(score_train(:,1:numpc),Gender,score_test(:,1:numpc));

%% a discriminative method
addpath('./liblinear');
[PredictedLabel] = logistic(Words, Gender , WordsTest);

%% An instance based method
PredictedLabel=KNearestNeighbor(Words,Gender,WordsTest,4);
%% regularization method
[NoisyWords]=AddNoise(Words,.1);
[PredictedLabel] = logistic(NoisyWords, Gender , WordsTest);
%% semi supervised dimensionality reduction of the data

[ score_train,score_test, numpc] = pca_getpc(Words,WordsTest);
[PredictedLabel] = logistic( score_train(:, 1:numpc), Gender, score_test(:,1:numpc));
function predictions = make_final_prediction(model,X_test)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% Sample model

score_test = double(X_test(:,1:5000)) * model.coeff_train;

addpath ./liblinear
[predictions] = predict(round(rand(size(score_test,1),1)), sparse(score_test), model, ['-q', 'col']);


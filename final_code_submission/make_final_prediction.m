function predictions = make_final_prediction(model,X_test)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% Sample model

%we only use words and image feats
X_test = [X_test(:,1:5000) X_test(:,end-6:end)];
X_test(:,model.featSel) = [];  %remove features from feature selection

avgX = model.Xstats(1,:);
stdX = model.Xstats(2,:);
Xnorm_test = bsxfun(@rdivide ,((X_test)  - repmat(avgX,size(X_test,1),1)), stdX);


yhat_ada   = (predict(model.ada,X_test));
yhat_dec   = str2num(cell2mat(predict(model.rndForest,X_test)));


[yhat_lin] = predict(round(rand(size(Xnorm_test,1),1)), sparse(Xnorm_test), model.liblin, ['-q', 'col']);
[yhat_svm] = svmpredict(ones(size(Xnorm_test,1),1), Xnorm_test, model.svm);


yhat_ens = [yhat_dec  yhat_lin  yhat_ada yhat_svm];
yhat_ens(yhat_ens == 0 ) = -1;   %make sure all are in the binary class -1/+1 instead of 0/1 for ensemble weights

predictions = sign(yhat_ens*model.wEns);
predictions(predictions == -1 ) = 0;  %convert back to 0/1

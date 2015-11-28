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


score_test=score_test(:,model.FeaturesWithVariance);

for Outcome=1:size(model.MeanFeatureOfLabel,1);
    for AFeature=1:size(score_test,2)
        ProbabilityOfFeatureGivenLabel(:,Outcome,AFeature)=normpdf(score_test(:,AFeature),model.MeanFeatureOfLabel(Outcome,AFeature),model.STDFeaturesOfLabel(Outcome,AFeature));
    end
end
    ProbabilityOfAllFeaturesGivenLabel=prod(ProbabilityOfFeatureGivenLabel,3);
    ProbabilityOfLetterGivenAllFeatures=(model.ProbabilityOfLabel'*ones(1,size(ProbabilityOfAllFeaturesGivenLabel,1)))'.*ProbabilityOfAllFeaturesGivenLabel;
    
    [~,PredictedLabel]=max(ProbabilityOfLetterGivenAllFeatures,[],2);
    predictions=PredictedLabel-1;






function model = init_model()

load('GNBModel.mat');
model.FeaturesWithVariance=FeaturesWithVariance;
model.coeff_train=coeff_train;
model.MeanFeatureOfLabel=MeanFeaturesOfLabel;
model.STDFeaturesOfLabel=STDFeaturesOfLabel;
model.ProbabilityOfLabel=ProbabilityOfLabel;

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];

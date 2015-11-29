function model = init_model()

load('SSDModel.mat')
model.coeff_train=coeff_train;

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];

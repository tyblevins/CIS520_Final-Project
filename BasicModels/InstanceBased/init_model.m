function model = init_model()

load('KNNModel.mat')
model.gender_train=gender_train;
model.words_train=words_train;
% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];

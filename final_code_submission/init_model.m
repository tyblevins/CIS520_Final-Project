function model = init_model()

load('adaLasso_mod.mat'); %cada
load('decLasso_mod.mat'); %rndForest
liblin = load('liblinLasso_mod.mat');
liblin = liblin.model;
svm = load('svmLasso_mod.mat');
svm = svm.mod;
load('wEns.mat');    %wEns
load('Xstats.mat');  %Xstats [avg std];

load('lassoFeatSel.mat')
featSel = (lassoResults{1} == 0);


model = struct('ada',cada,'rndForest',rndForest,'liblin',liblin,'svm',svm,'featSel',featSel,'wEns',wEns,'Xstats',Xstats);


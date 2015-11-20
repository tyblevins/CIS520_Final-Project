%Jared Wilson 
%ensemble attempt
clear;

yhat_knn  = load('yhat_knn.mat');
yhat_knn = yhat_knn.yhat_knn;
yhat_dec  = load('yhat_decTree');
yhat_dec = yhat_dec.yhat;
yhat_perc = load('yhat_perc.mat');
yhat_perc = yhat_perc.yhat;


yhat_ens = [yhat_knn yhat_dec yhat_perc];

yhat_tot = mode(yhat_ens,2);

dlmwrite('submit.txt', yhat_tot);

dlmwrite('submit.txt', yhat_dec);

%create confusion matrix to show percent aggree between each model
yhat_NEG = yhat_ens;
yhat_NEG(yhat_ens == 0) = -1;

agreeMAT = yhat_NEG'*yhat_NEG;

agreeMAT = abs(agreeMAT - length(yhat_knn))./2;

figure(1)
imagesc(agreeMAT);
colorbar;
set(gca,'XTick',[1:3])
set(gca,'YTick',[1:3])
set(gca,'XTickLabel',{'K-NN','DecTree', 'Perceptron'})
set(gca,'YTickLabel',{'K-NN','DecTree', 'Perceptron'})

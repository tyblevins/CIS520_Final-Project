
[coeff_train, latent] = pcacov(words_train'*words_train);
coeff_train=coeff_train(:,1:50);
score_train = words_train * coeff_train;


addpath ./liblinear
model = train(gender_train, sparse(score_train), ['-s 0', 'col']);
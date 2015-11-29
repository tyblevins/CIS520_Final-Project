[coeff_train, latent] = pcacov(words_train'*words_train);
coeff_train=coeff_train(:,1:50);
score_train = words_train * coeff_train;

FeaturesWithVariance=var(score_train)~=0;
words_train=words_train(:,FeaturesWithVariance);

for Outcome=1:length(unique(gender_train))
        ProbabilityOfLabel(Outcome)=sum(gender_train==Outcome-1)/numel(gender_train); 
        FeaturesOfLabel=words_train(gender_train==Outcome-1,:);
        MeanFeaturesOfLabel(Outcome,:)=mean(FeaturesOfLabel); 
        STDFeaturesOfLabel(Outcome,:)=std(FeaturesOfLabel); 
end